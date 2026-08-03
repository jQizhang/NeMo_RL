# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training utilities for automodel (DTensor-based) policy workers.

This module provides post-processor classes and forward/backward functions
that follow the same pattern as nemo_rl/models/megatron/train.py.

Key differences from megatron approach:
- Post-processors compute results directly (no callable return pattern)
- forward_with_post_processing_fn calls post-processor directly
- automodel_forward_backward uses PyTorch autograd instead of Megatron's pipeline
"""

import enum
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import torch
from nemo_automodel.components.distributed.context_parallel import (
    ContextParallelSharder,
)
from nemo_automodel.components.distributed.tensor_utils import to_local_if_dtensor
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss import SequencePackingLossWrapper, prepare_loss_input
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.algorithms.loss.utils import (
    needs_unfiltered_reference_logprobs,
    prepare_precomputed_distillation_loss_input,
    prepare_precomputed_logprob_loss_input,
)
from nemo_rl.algorithms.x_token.loss_utils import (
    prepare_xtoken_window_loss_inputs,
)
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    distributed_vocab_topk,
    get_student_distillation_statistics_from_vocab_parallel_logits,
    get_target_logprobs_from_vocab_parallel_logits,
    resolve_vocab_parallel_logits,
)
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.policy import PolicyConfig

# Union type for any post-processing function
PostProcessingFunction = Union[
    "LossPostProcessor",
    "LogprobsPostProcessor",
    "TopkLogitsPostProcessor",
    "FullLogitsPostProcessor",
    "ScorePostProcessor",
]


class CPLossContract(enum.Enum):
    """How an Automodel loss consumes model contributions across CP ranks."""

    REPLICATED = "replicated"
    PARTITIONED = "partitioned"


_CP_LOSS_CONTRACTS = {
    LossInputType.LOGPROB: CPLossContract.REPLICATED,
    LossInputType.DISTILLATION: CPLossContract.REPLICATED,
    LossInputType.DISTILLATION_CROSS_TOKENIZER: CPLossContract.PARTITIONED,
}


@dataclass
class PreparedModelForward:
    """Model inputs and Automodel layout resolved for one microbatch."""

    model_batch: dict[str, Any]
    cp_sharder: ContextParallelSharder
    model_context_factory: Callable[[], AbstractContextManager[Any]]


@dataclass(frozen=True)
class FullLogitsShard:
    """One producer's contiguous sequence and vocabulary logits shard."""

    local_logits: torch.Tensor
    tp_rank: int
    tp_size: int
    vocab_start_index: int
    vocab_end_index: int
    full_vocab_size: int
    global_seq_start: int
    full_seq_len: int
    vocab_sharded: bool
    sequence_sharded: bool


def get_model_output_vocab_size(model: nn.Module) -> int:
    """Return the logical vocabulary width emitted by the model head."""
    output_head = None
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        output_head = get_output_embeddings()
    if output_head is None:
        output_head = getattr(model, "lm_head", None)

    head_width: Optional[int] = None
    if output_head is not None:
        out_features = getattr(output_head, "out_features", None)
        if out_features is not None:
            head_width = int(out_features)
        weight = getattr(output_head, "weight", None)
        if weight is not None:
            weight_width = int(weight.shape[0])
            if head_width is not None and weight_width != head_width:
                raise ValueError(
                    "The output head reports inconsistent vocabulary widths: "
                    f"out_features={head_width}, weight.shape[0]={weight_width}."
                )
            head_width = weight_width

    config_width = getattr(getattr(model, "config", None), "vocab_size", None)
    if head_width is None and config_width is not None:
        head_width = int(config_width)
    if head_width is None or head_width <= 0:
        raise ValueError(
            "Could not resolve a positive logical output vocabulary width from "
            "the model output head or config."
        )
    return head_width


def _build_model_batch(
    model: nn.Module,
    processed_inputs: ProcessedInputs,
    *,
    is_reward_model: bool,
    allow_flash_attn_args: bool,
) -> dict[str, Any]:
    """Build a private model-facing batch for one forward."""
    model_batch: dict[str, Any] = {
        "input_ids": processed_inputs.input_ids,
        "use_cache": False,
    }
    if processed_inputs.attention_mask is not None:
        model_batch["attention_mask"] = processed_inputs.attention_mask
    if processed_inputs.position_ids is not None:
        model_batch["position_ids"] = processed_inputs.position_ids
    if processed_inputs.has_flash_attention:
        model_batch["flash_attn_kwargs"] = processed_inputs.flash_attn_kwargs

    if processed_inputs.is_multimodal:
        reserved = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "use_cache",
            "flash_attn_kwargs",
            "labels",
        }
        collisions = reserved.intersection(processed_inputs.vlm_kwargs)
        if collisions:
            raise ValueError(
                "Multimodal kwargs collide with model-batch-owned keys: "
                f"{sorted(collisions)}."
            )
        model_batch.update(processed_inputs.vlm_kwargs)
        model_batch.pop("flash_attn_kwargs", None)

    is_gemma3 = isinstance(model, Gemma3ForCausalLM) or isinstance(
        model, Gemma3ForConditionalGeneration
    )
    if is_gemma3 and "token_type_ids" not in model_batch:
        model_batch["token_type_ids"] = torch.zeros_like(processed_inputs.input_ids)

    if getattr(getattr(model, "config", None), "model_type", None) == "gemma4":
        if "mm_token_type_ids" not in model_batch:
            model_batch["mm_token_type_ids"] = torch.zeros_like(
                processed_inputs.input_ids
            )

    if is_reward_model or not allow_flash_attn_args:
        model_batch.pop("flash_attn_kwargs", None)

    for key in (
        "input_ids",
        "position_ids",
        "attention_mask",
        "token_type_ids",
        "mm_token_type_ids",
    ):
        value = model_batch.get(key)
        if isinstance(value, torch.Tensor):
            model_batch[key] = value.clone()
    return model_batch


def prepare_model_forward(
    model: nn.Module,
    processed_inputs: ProcessedInputs,
    *,
    device_mesh: Optional[DeviceMesh],
    padding_token_id: int,
    is_reward_model: bool,
    allow_flash_attn_args: bool,
) -> PreparedModelForward:
    """Build model inputs and resolve Automodel CP for one microbatch."""
    model_batch = _build_model_batch(
        model,
        processed_inputs,
        is_reward_model=is_reward_model,
        allow_flash_attn_args=allow_flash_attn_args,
    )
    if "labels" in model_batch:
        raise ValueError("NeMo RL model batches must not provide model-owned labels.")
    input_ids = model_batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("The Automodel NeMo RL path requires tensor input_ids.")
    model_batch["labels"] = torch.full_like(input_ids, -100)

    cp_sharder = ContextParallelSharder(
        model,
        device_mesh,
        model_batch,
        padding_token_id=padding_token_id,
        num_chunks=1,
    )
    model_context_factory, model_batch = cp_sharder.shard(model_batch)
    model_batch.pop("labels")
    return PreparedModelForward(
        model_batch=model_batch,
        cp_sharder=cp_sharder,
        model_context_factory=model_context_factory,
    )


def model_forward(
    model: nn.Module,
    model_batch: dict[str, Any],
) -> Any:
    """Run a model on an already prepared model-facing batch.

    Args:
        model: The model to run forward pass on
        model_batch: Private batch already processed by the Automodel sharder.

    Returns:
        The model-specific forward output.
    """
    return model(**model_batch)


def extract_logits(
    model: nn.Module,
    outputs: Any,
) -> torch.Tensor:
    """Extract logits from model outputs.

    Args:
        model: The model (used for lm_head if needed)
        outputs: Model outputs (can be tensor, DTensor, or object with logits attribute)

    Returns:
        torch.Tensor: Logits tensor
    """
    if isinstance(outputs, (torch.Tensor, DTensor)):
        # Custom models can output logits directly
        return outputs
    elif not hasattr(outputs, "logits"):
        return model.lm_head(outputs.last_hidden_state)
    else:
        return outputs.logits


def apply_temperature_scaling(
    logits: torch.Tensor, sampling_params: Optional[TrainingSamplingParams]
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Logits tensor to scale
        sampling_params: Sampling parameters

    Returns:
        torch.Tensor: Temperature-scaled logits
    """
    if sampling_params is not None and sampling_params.temperature != 1.0:
        logits.div_(sampling_params.temperature)
    return logits


def forward_with_post_processing_fn(
    model: nn.Module,
    prepared: PreparedModelForward,
    post_processing_fn: PostProcessingFunction,
    processed_mb: ProcessedMicrobatch,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    sequence_dim: int = 1,
) -> Tuple[Any, dict[str, Any], ProcessedMicrobatch]:
    """Perform forward pass with pre-processed microbatch and apply post-processing.

    This function takes a pre-processed microbatch (with sequence packing already handled),
    runs the forward step through the model, and applies the post-processing function
    to compute the result.

    Unlike the megatron approach which returns a callable, this directly computes
    and returns the result since automodel uses PyTorch autograd.

    Args:
        model: The model to run forward pass on
        prepared: Per-microbatch model batch, context, and token layout.
        post_processing_fn: Post-processing function to apply to the logits
        processed_mb: Pre-fetched ProcessedMicrobatch containing data and processed inputs
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        sequence_dim: Sequence dimension

    Returns:
        tuple: (result, metrics, processed_microbatch)
            - result: Output from post-processing (loss, logprobs, topk, or scores)
            - metrics: Dictionary of metrics from post-processing
            - processed_microbatch: The ProcessedMicrobatch that was processed
    """
    # Extract the processed components
    data_dict = processed_mb.data_dict
    processed_inputs = processed_mb.processed_inputs

    # Model forward pass
    outputs = model_forward(model, prepared.model_batch)

    # Extract logits from model outputs
    logits = extract_logits(model, outputs)
    del outputs

    # Apply temperature scaling only for sampling-oriented post-processors
    # Score computations should use unscaled logits
    if isinstance(
        post_processing_fn,
        (
            LossPostProcessor,
            LogprobsPostProcessor,
            TopkLogitsPostProcessor,
            FullLogitsPostProcessor,
        ),
    ):
        # Temperature scaling is element-wise, directly applying it here.
        # Other sampling parameters like top-k and top-p need the logits from whole vocabulary,
        # so applying them when gathering logits from vocab parallel (called in LossPostProcessor and LogprobsPostProcessor).
        logits = apply_temperature_scaling(logits, sampling_params)

    # Apply the post-processing function directly based on type
    if isinstance(post_processing_fn, LossPostProcessor):
        result, metrics = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            cp_sharder=prepared.cp_sharder,
            sequence_dim=sequence_dim,
        )
    elif isinstance(
        post_processing_fn,
        (LogprobsPostProcessor, TopkLogitsPostProcessor),
    ):
        result = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=processed_mb.original_batch_size,
            original_seq_len=processed_mb.original_seq_len,
            cp_sharder=prepared.cp_sharder,
            sequence_dim=sequence_dim,
        )
        if isinstance(post_processing_fn, LogprobsPostProcessor):
            metrics = {"logprobs": result}
        else:
            vals, idx = result
            metrics = {"topk_logits": vals, "topk_indices": idx}
    elif isinstance(post_processing_fn, FullLogitsPostProcessor):
        result = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=processed_mb.original_batch_size,
            original_seq_len=processed_mb.original_seq_len,
            cp_sharder=prepared.cp_sharder,
            sequence_dim=sequence_dim,
        )
        metrics = {"full_logits": result.local_logits}
    elif isinstance(post_processing_fn, ScorePostProcessor):
        result = post_processing_fn(logits=logits)
        metrics = {"scores": result}
    else:
        raise TypeError(
            f"Unknown post-processing function type: {type(post_processing_fn)}"
        )

    del logits
    return result, metrics, processed_mb


def automodel_forward_backward(
    model: nn.Module,
    data_iterator: Iterator[ProcessedMicrobatch],
    post_processing_fn: PostProcessingFunction,
    device_mesh: Optional[DeviceMesh],
    padding_token_id: int,
    autocast_context_factory: Callable[[], AbstractContextManager[Any]],
    forward_only: bool = False,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    sequence_dim: int = 1,
    dp_size: int = 1,
    cp_size: int = 1,
    num_global_batches: int = 1,
    num_valid_microbatches: Optional[int] = None,
    on_microbatch_start: Optional[Callable[[int], None]] = None,
) -> list[Tuple[Any, dict[str, Any]]]:
    """Execute forward and backward passes for automodel.

    This is the main training loop function that coordinates forward and backward
    passes across multiple microbatches using PyTorch autograd.

    Unlike megatron_forward_backward which uses Megatron's pipeline parallel
    framework, this uses standard PyTorch operations.

    Args:
        model: The model to train
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        post_processing_fn: Post-processing function to apply to the logits
        device_mesh: Worker device mesh used by Automodel CP resolution.
        padding_token_id: Token ID used for Automodel sequence padding.
        autocast_context_factory: Worker-owned precision context factory.
        forward_only: If True, skip backward pass
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        sequence_dim: Sequence dimension
        dp_size: Data parallel size
        cp_size: Context parallel size
        num_global_batches: Number of global batches (for metric scaling)
        num_valid_microbatches: Number of valid (non-dummy) microbatches. If provided,
            microbatches beyond this index are treated as dummy batches (loss *= 0).
            If None, all microbatches are considered valid.
        on_microbatch_start: Optional callback called at the start of each microbatch
            with the microbatch index. Useful for cache clearing, etc.

    Returns:
        List of (result, metrics) tuples from each microbatch
    """
    if not forward_only and not isinstance(post_processing_fn, LossPostProcessor):
        raise TypeError("Backward execution requires LossPostProcessor.")
    results = []

    for mb_idx, processed_mb in enumerate(data_iterator):
        # Call optional callback at start of microbatch
        if on_microbatch_start is not None:
            on_microbatch_start(mb_idx)

        processed_inputs = processed_mb.processed_inputs
        prepared = prepare_model_forward(
            model,
            processed_inputs,
            device_mesh=device_mesh,
            padding_token_id=padding_token_id,
            is_reward_model=is_reward_model,
            allow_flash_attn_args=allow_flash_attn_args,
        )

        with prepared.model_context_factory(), autocast_context_factory():
            # Forward pass with post-processing
            result, metrics, _ = forward_with_post_processing_fn(
                model=model,
                prepared=prepared,
                post_processing_fn=post_processing_fn,
                processed_mb=processed_mb,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                sampling_params=sampling_params,
                sequence_dim=sequence_dim,
            )

            # Check if this is a dummy batch
            is_dummy = (
                num_valid_microbatches is not None and mb_idx >= num_valid_microbatches
            )

            # Scale metrics for aggregation (only for loss)
            if isinstance(post_processing_fn, LossPostProcessor):
                # skip the update for dummy batches
                if not is_dummy:
                    ## scale by the number of global batches so we get the correct
                    ## value when summing metrics across all microbatches
                    for k in metrics.keys():
                        if "_min" in k or "_max" in k:
                            continue

                        metrics[k] /= num_global_batches
                else:
                    # Zero out loss for dummy batches
                    result = result * 0

                # Backward pass if training
                if not forward_only:
                    ## NOTE: invalid samples should be multiplied
                    ## by zero in the loss function to prevent them
                    ## from affecting the gradient calculation

                    # when FSDP reduces the gradients over the DP dim, they're automatically averaged
                    # but we want to sum them so we cancel out the average here
                    loss = (
                        result
                        * dp_size
                        * cp_size
                        / post_processing_fn.cp_gradient_fanout
                    )
                    loss.backward()

        results.append((result, metrics))

    return results


class LossPostProcessor:
    """Post-processor for computing training loss from model outputs."""

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        tp_mesh: DeviceMesh,
        expected_global_vocab_size: Optional[int],
        padding_token_id: int,
        cp_size: int,
        dp_size: int,
        cp_group: Optional[torch.distributed.ProcessGroup],
        dp_group: Optional[torch.distributed.ProcessGroup],
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ) -> None:
        """Initialize LossPostProcessor.

        Args:
            loss_fn: Loss function to compute loss
            cfg: Configuration dictionary
            tp_mesh: Tensor parallel mesh
            expected_global_vocab_size: Logical model output vocabulary width.
            cp_size: Context parallel size
            dp_size: Data parallel size
            cp_group: Exact context-parallel group, or None at CP1.
            dp_group: Exact data-parallel group, or None at DP1.
            enable_seq_packing: Whether sequence packing is enabled
            sampling_params: Sampling parameters
        """
        self.loss_fn: LossFunction = loss_fn
        self.cfg: PolicyConfig = cfg
        self.tp_mesh = tp_mesh
        self.expected_global_vocab_size = expected_global_vocab_size
        self.padding_token_id = padding_token_id
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.cp_group = cp_group
        self.dp_group = dp_group
        self.enable_seq_packing = enable_seq_packing
        self.sampling_params = sampling_params
        self.logprob_chunk_size = cfg.get("logprob_chunk_size", None)

        if cp_size > 1:
            if (
                cp_group is None
                or torch.distributed.get_world_size(cp_group) != cp_size
            ):
                raise ValueError(
                    "cp_group must match cp_size for CP-enabled loss processing."
                )
        elif cp_group is not None:
            raise ValueError("cp_group must be None when cp_size == 1.")
        if dp_size > 1:
            if (
                dp_group is None
                or torch.distributed.get_world_size(dp_group) != dp_size
            ):
                raise ValueError(
                    "dp_group must match dp_size for distributed loss processing."
                )
        elif dp_group is not None:
            raise ValueError("dp_group must be None when dp_size == 1.")

        self._cp_contract = _CP_LOSS_CONTRACTS.get(loss_fn.input_type)
        if cp_size > 1 and self._cp_contract is None:
            raise ValueError(
                f"CP>1 does not support loss input type {loss_fn.input_type}."
            )
        self._cp_gradient_fanout = (
            cp_size
            if cp_size > 1 and self._cp_contract is CPLossContract.REPLICATED
            else 1
        )

    def _require_output_vocab_size(self) -> int:
        """Return the configured model vocabulary for vocabulary-based losses."""
        if self.expected_global_vocab_size is None:
            raise ValueError(
                f"Loss input type {self.loss_fn.input_type} requires a logical "
                "model output vocabulary size."
            )
        return self.expected_global_vocab_size

    @property
    def cp_gradient_fanout(self) -> int:
        """Number of CP loss consumers for each local model contribution."""
        return self._cp_gradient_fanout

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        *,
        cp_sharder: ContextParallelSharder,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss from logits.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            global_valid_seqs: Global valid sequence count
            global_valid_toks: Global valid token count
            cp_sharder: Per-microbatch Automodel token-layout owner.
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (loss, metrics)
        """
        del sequence_dim
        prepare_loss_input_wrapped = partial(
            prepare_loss_input, sampling_params=self.sampling_params
        )
        if self.enable_seq_packing:
            if self.cp_size != 1:
                raise ValueError("Sequence packing is not supported with CP>1.")
            loss_fn = SequencePackingLossWrapper(
                loss_fn=self.loss_fn,
                prepare_fn=prepare_loss_input_wrapped,
                cu_seqlens_q=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
                cu_seqlens_q_padded=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
            )
            loss, loss_metrics = loss_fn(
                logits,
                data_dict,
                global_valid_seqs,
                global_valid_toks,
            )
            return loss, loss_metrics

        if self.loss_fn.input_type == LossInputType.LOGPROB:
            vocab = resolve_vocab_parallel_logits(
                logits,
                tp_mesh=self.tp_mesh,
                expected_global_vocab_size=self._require_output_vocab_size(),
            )
            input_ids = data_dict["input_ids"].to(vocab.local_logits.device)
            targets = input_ids.roll(shifts=-1, dims=1)
            local_targets = cp_sharder.shard_token_tensor(
                targets, seq_dim=1, fill=self.padding_token_id
            )
            local_logprobs = get_target_logprobs_from_vocab_parallel_logits(
                vocab.local_logits,
                local_targets,
                tp_group=vocab.vocab_parallel_group,
                vocab_start_index=vocab.vocab_start_index,
                vocab_end_index=vocab.vocab_end_index,
                global_vocab_size=vocab.global_vocab_size,
                chunk_size=self.logprob_chunk_size,
                sampling_params=self.sampling_params,
                inference_only=False,
            )
            logprobs = cp_sharder.gather_token_tensor(
                local_logprobs, seq_dim=1, trim=True, fill=0.0
            )[:, :-1]

            unfiltered_logprobs = None
            if needs_unfiltered_reference_logprobs(self.loss_fn, self.sampling_params):
                local_unfiltered = get_target_logprobs_from_vocab_parallel_logits(
                    vocab.local_logits,
                    local_targets,
                    tp_group=vocab.vocab_parallel_group,
                    vocab_start_index=vocab.vocab_start_index,
                    vocab_end_index=vocab.vocab_end_index,
                    global_vocab_size=vocab.global_vocab_size,
                    chunk_size=self.logprob_chunk_size,
                    sampling_params=None,
                    inference_only=False,
                )
                unfiltered_logprobs = cp_sharder.gather_token_tensor(
                    local_unfiltered, seq_dim=1, trim=True, fill=0.0
                )[:, :-1]

            loss_input, data_dict = prepare_precomputed_logprob_loss_input(
                logprobs,
                data_dict,
                self.loss_fn,
                sampling_params=self.sampling_params,
                unfiltered_logprobs=unfiltered_logprobs,
            )
        elif self.loss_fn.input_type == LossInputType.DISTILLATION:
            vocab = resolve_vocab_parallel_logits(
                logits,
                tp_mesh=self.tp_mesh,
                expected_global_vocab_size=self._require_output_vocab_size(),
            )
            teacher_indices = data_dict["teacher_topk_indices"].to(
                vocab.local_logits.device
            )
            local_teacher_indices = cp_sharder.shard_token_tensor(
                teacher_indices, seq_dim=1, fill=0
            )
            calculate_entropy = (
                self.loss_fn.zero_outside_topk and self.loss_fn.kl_type != "forward"
            )
            local_student_logprobs, local_entropy = (
                get_student_distillation_statistics_from_vocab_parallel_logits(
                    vocab.local_logits,
                    local_teacher_indices,
                    tp_group=vocab.vocab_parallel_group,
                    vocab_start_index=vocab.vocab_start_index,
                    vocab_end_index=vocab.vocab_end_index,
                    global_vocab_size=vocab.global_vocab_size,
                    zero_outside_topk=self.loss_fn.zero_outside_topk,
                    calculate_entropy=calculate_entropy,
                    chunk_size=self.logprob_chunk_size,
                )
            )
            student_logprobs = cp_sharder.gather_token_tensor(
                local_student_logprobs, seq_dim=1, trim=True, fill=0.0
            )
            entropy = None
            if local_entropy is not None:
                entropy = cp_sharder.gather_token_tensor(
                    local_entropy, seq_dim=1, trim=True, fill=0.0
                )
            loss_input = prepare_precomputed_distillation_loss_input(
                student_logprobs,
                data_dict["teacher_topk_logits"],
                entropy,
            )
        elif self.loss_fn.input_type == LossInputType.DISTILLATION_CROSS_TOKENIZER:
            vocab = resolve_vocab_parallel_logits(
                logits,
                tp_mesh=self.tp_mesh,
                expected_global_vocab_size=self._require_output_vocab_size(),
            )
            full_student_logits = cp_sharder.gather_token_tensor(
                vocab.local_logits, seq_dim=1, trim=True, fill=0.0
            )
            loss_input = prepare_xtoken_window_loss_inputs(
                full_student_logits,
                data_dict,
                vocab=vocab,
                student_tokenizer_vocab_size=self.loss_fn.student_tokenizer_vocab_size,
                teacher_tokenizer_vocab_sizes=self.loss_fn.teacher_vocab_sizes,
                projection_matrix_paths=self.loss_fn.projection_matrix_paths,
                context_parallel_group=self.cp_group,
                data_parallel_group=self.dp_group,
                logprob_chunk_size=self.logprob_chunk_size,
            )
            del full_student_logits
        else:
            loss_input, data_dict = prepare_loss_input_wrapped(
                logits, data_dict, self.loss_fn
            )

        loss, loss_metrics = self.loss_fn(
            data=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            **loss_input,
        )
        return loss, loss_metrics


class LogprobsPostProcessor:
    """Post-processor for computing log probabilities from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
        tp_mesh: DeviceMesh,
        expected_global_vocab_size: int,
        padding_token_id: int,
        cp_size: int,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ) -> None:
        """Initialize the logprob postprocessor."""
        self.cfg = cfg
        self.tp_mesh = tp_mesh
        self.expected_global_vocab_size = expected_global_vocab_size
        self.padding_token_id = padding_token_id
        self.cp_size = cp_size
        self.enable_seq_packing = enable_seq_packing
        self.sampling_params = sampling_params
        self.logprob_chunk_size = cfg.get("logprob_chunk_size", None)

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        *,
        cp_sharder: ContextParallelSharder,
        sequence_dim: int = 1,
    ) -> torch.Tensor:
        """Compute token log probabilities from logits.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            sequence_dim: Sequence dimension

        Returns:
            Token log probabilities tensor [batch_size, seq_length]
        """
        del sequence_dim
        seq_len = processed_inputs.seq_len
        input_lengths = data_dict["input_lengths"]
        vocab = resolve_vocab_parallel_logits(
            logits,
            tp_mesh=self.tp_mesh,
            expected_global_vocab_size=self.expected_global_vocab_size,
        )
        source_input_ids = (
            processed_inputs.input_ids
            if self.enable_seq_packing
            else data_dict["input_ids"]
        ).to(vocab.local_logits.device)
        targets = source_input_ids.roll(shifts=-1, dims=1)
        local_targets = cp_sharder.shard_token_tensor(
            targets, seq_dim=1, fill=self.padding_token_id
        )
        local_logprobs = get_target_logprobs_from_vocab_parallel_logits(
            vocab.local_logits,
            local_targets,
            tp_group=vocab.vocab_parallel_group,
            vocab_start_index=vocab.vocab_start_index,
            vocab_end_index=vocab.vocab_end_index,
            global_vocab_size=vocab.global_vocab_size,
            chunk_size=self.logprob_chunk_size,
            sampling_params=self.sampling_params,
            inference_only=True,
        )
        token_logprobs = cp_sharder.gather_token_tensor(
            local_logprobs, seq_dim=1, trim=True, fill=0.0
        )[:, :-1]

        # Prepend 0 for first token to maintain sequence length
        token_logprobs = torch.cat(
            [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
        )

        # Handle sequence packing unpacking or mask application
        if self.enable_seq_packing:
            unpacked_logprobs = torch.zeros(
                (original_batch_size, original_seq_len),
                dtype=token_logprobs.dtype,
                device=token_logprobs.device,
            )
            cu_seqlens = processed_inputs.flash_attn_kwargs.cu_seqlens_q
            for i in range(original_batch_size):
                start = cu_seqlens[i].item() + 1
                end = cu_seqlens[i + 1].item()
                seq_len_actual = input_lengths[i].item()
                unpacked_logprobs[i, 1:seq_len_actual] = token_logprobs[0, start:end]
            token_logprobs = unpacked_logprobs
        else:
            # Apply mask to zero out padding tokens logprobs
            batch_size = processed_inputs.input_ids.shape[0]
            post_attention_mask = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=token_logprobs.device,
            )
            for i, length in enumerate(input_lengths):
                # For right-padded sequence, set 1s at the beginning of the sequence
                post_attention_mask[i, :length] = 1
            token_logprobs = token_logprobs * post_attention_mask

        # handle top-k/top-p filtering for logprobs, only used for ClippedPGLossFn now
        if need_top_k_or_top_p_filtering(self.sampling_params):
            mask = data_dict["token_mask"] * data_dict["sample_mask"].unsqueeze(-1)
            token_logprobs = mask_out_neg_inf_logprobs(
                token_logprobs, mask, "prev_logprobs"
            )

        return token_logprobs


class TopkLogitsPostProcessor:
    """Post-processor for computing top-k logits from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
        tp_mesh: DeviceMesh,
        expected_global_vocab_size: int,
        cp_size: int,
        k: int,
        enable_seq_packing: bool = False,
    ) -> None:
        """Initialize TopkLogitsPostProcessor.

        Args:
            cfg: Configuration dictionary
            tp_mesh: Tensor parallel mesh
            expected_global_vocab_size: Logical model output vocabulary width.
            cp_size: Context parallel size
            k: Number of top logits to return
            enable_seq_packing: Whether sequence packing is enabled
        """
        self.cfg = cfg
        self.tp_mesh = tp_mesh
        self.expected_global_vocab_size = expected_global_vocab_size
        self.cp_size = cp_size
        self.k = k
        self.enable_seq_packing = enable_seq_packing

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        *,
        cp_sharder: ContextParallelSharder,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k logits and indices from model outputs.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (top-k values, top-k indices) tensors
        """
        del sequence_dim
        input_lengths = data_dict["input_lengths"]
        vocab = resolve_vocab_parallel_logits(
            logits,
            tp_mesh=self.tp_mesh,
            expected_global_vocab_size=self.expected_global_vocab_size,
        )
        if not 1 <= self.k <= vocab.global_vocab_size:
            raise ValueError(
                f"k must be in [1, {vocab.global_vocab_size}], got {self.k}."
            )
        if vocab.vocab_parallel_group is not None:
            vals, idx = distributed_vocab_topk(
                vocab.local_logits,
                k=self.k,
                tp_group=vocab.vocab_parallel_group,
                vocab_start_index=vocab.vocab_start_index,
                vocab_end_index=vocab.vocab_end_index,
                global_vocab_size=vocab.global_vocab_size,
            )
        else:
            vals, idx = torch.topk(
                vocab.local_logits.to(torch.float32), k=self.k, dim=-1
            )
        vals = cp_sharder.gather_token_tensor(vals, seq_dim=1, trim=True, fill=0.0)
        idx = cp_sharder.gather_token_tensor(idx, seq_dim=1, trim=True, fill=0)

        # Handle sequence packing unpacking
        if self.enable_seq_packing:
            # Unpack top-k results from packed format back to original batch format
            # vals: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
            # idx: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
            unpacked_vals = torch.zeros(
                (original_batch_size, original_seq_len, self.k),
                dtype=vals.dtype,
                device=vals.device,
            )
            unpacked_idx = torch.zeros(
                (original_batch_size, original_seq_len, self.k),
                dtype=idx.dtype,
                device=idx.device,
            )

            cu_seqlens = processed_inputs.flash_attn_kwargs.cu_seqlens_q

            for i in range(original_batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_len_actual = input_lengths[i].item()

                # Extract the corresponding portion from packed results
                # Note: vals and idx are [1, packed_seq_len, k] due to packing
                unpacked_vals[i, :seq_len_actual, :] = vals[0, start:end, :]
                unpacked_idx[i, :seq_len_actual, :] = idx[0, start:end, :]

            vals = unpacked_vals
            idx = unpacked_idx

        return vals, idx


class FullLogitsPostProcessor:
    """Export one teacher's contiguous CP window and local vocabulary shard.

    Automodel first restores model-layout logits to canonical token order. This
    postprocessor retains only the producer's contiguous CP window; the IPC
    consumer reassembles the requested full-vocabulary window from all producer
    rectangles. Teacher and student TP/CP topologies may differ. Sequence
    packing is unsupported.
    """

    def __init__(
        self,
        tp_mesh: DeviceMesh,
        expected_global_vocab_size: int,
        producer_cp_rank: int,
        producer_cp_size: int,
        enable_seq_packing: bool = False,
    ) -> None:
        self.tp_mesh = tp_mesh
        self.expected_global_vocab_size = expected_global_vocab_size
        self.producer_cp_rank = producer_cp_rank
        self.producer_cp_size = producer_cp_size
        self.enable_seq_packing = enable_seq_packing
        if producer_cp_size <= 0 or not 0 <= producer_cp_rank < producer_cp_size:
            raise ValueError(
                "Invalid producer CP topology: "
                f"rank={producer_cp_rank}, size={producer_cp_size}."
            )

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: Any,
        original_batch_size: int,
        original_seq_len: int,
        *,
        cp_sharder: ContextParallelSharder,
        sequence_dim: int = 1,
    ) -> FullLogitsShard:
        del data_dict, processed_inputs, sequence_dim
        if self.enable_seq_packing:
            raise NotImplementedError(
                "FullLogitsPostProcessor: sequence packing is not supported in v0."
            )
        vocab = resolve_vocab_parallel_logits(
            logits,
            tp_mesh=self.tp_mesh,
            expected_global_vocab_size=self.expected_global_vocab_size,
        )
        local_logits = vocab.local_logits.to(torch.float32)
        full_logits = cp_sharder.gather_token_tensor(
            local_logits, seq_dim=1, trim=True, fill=0.0
        )
        if full_logits.shape[0] != original_batch_size:
            raise ValueError(
                "Restored full logits have the wrong batch size: "
                f"expected {original_batch_size}, got {full_logits.shape[0]}."
            )
        full_sequence_length = int(full_logits.shape[1])
        if full_sequence_length != original_seq_len:
            raise ValueError(
                "Restored full logits do not match the canonical microbatch length: "
                f"expected {original_seq_len}, got {full_sequence_length}."
            )
        if full_sequence_length % self.producer_cp_size != 0:
            raise ValueError(
                "Full-logits IPC requires equal producer CP windows, got "
                f"sequence length {full_sequence_length} and CP size "
                f"{self.producer_cp_size}."
            )
        local_sequence_length = full_sequence_length // self.producer_cp_size
        global_seq_start = self.producer_cp_rank * local_sequence_length
        global_seq_end = global_seq_start + local_sequence_length
        contiguous_logits = full_logits[
            :, global_seq_start:global_seq_end, :
        ].contiguous()
        del full_logits
        return FullLogitsShard(
            local_logits=contiguous_logits,
            tp_rank=vocab.tp_rank,
            tp_size=vocab.tp_size,
            vocab_start_index=vocab.vocab_start_index,
            vocab_end_index=vocab.vocab_end_index,
            full_vocab_size=vocab.global_vocab_size,
            global_seq_start=global_seq_start,
            full_seq_len=full_sequence_length,
            vocab_sharded=vocab.is_vocab_sharded,
            sequence_sharded=self.producer_cp_size > 1,
        )


class ScorePostProcessor:
    """Post-processor for computing reward model scores from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
    ):
        """Initialize ScorePostProcessor.

        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Extract scores from reward model outputs.

        Args:
            logits: Model output logits

        Returns:
            Scores tensor
        """
        logits = logits.to(torch.float32)
        rm_scores = to_local_if_dtensor(logits)
        rm_scores = rm_scores.squeeze(-1)

        return rm_scores


def aggregate_training_statistics(
    losses: list[float],
    all_mb_metrics: list[dict[str, Any]],
    grad_norm: Optional[torch.Tensor],
    dp_group: Any,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Aggregate training statistics across microbatches and ranks.

    Args:
        losses: List of loss values from each microbatch
        all_mb_metrics: List of metrics dictionaries from each microbatch
        grad_norm: Gradient norm tensor (or None if eval mode)
        dp_group: Data parallel process group for all-reduce
        dtype: Model dtype for metrics

    Returns:
        Dictionary containing aggregated metrics including global_loss, grad_norm, etc.
    """
    # Compute global loss across all ranks
    with torch.no_grad():
        global_loss = torch.tensor(losses, device="cuda")
        torch.distributed.all_reduce(global_loss, group=dp_group)

    # Aggregate metrics across all microbatches
    mb_metrics = defaultdict(list)
    for m in all_mb_metrics:
        for k, v in m.items():
            mb_metrics[k].append(v)

    metrics = {
        "global_loss": global_loss.cpu(),
        "grad_norm": grad_norm,
        "rank": torch.distributed.get_rank(),
        "gpu_name": torch.cuda.get_device_name(),
        "model_dtype": dtype,
        "all_mb_metrics": dict(mb_metrics),
    }

    return metrics
