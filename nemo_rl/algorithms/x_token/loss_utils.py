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
"""Shared utilities for cross-tokenizer distillation.

Used by both :mod:`token_aligner` and
:mod:`nemo_rl.algorithms.loss.loss_functions`:

- :class:`Fp32SparseMM` — FP32 sparse-dense matmul that ignores BF16
  autocast (no BF16 sparse-mm kernel exists).
- Chunk aggregation: :func:`chunk_log_prob_sums` / :func:`chunk_average_finalize`
  / :func:`chunk_average_log_probs` / :func:`valid_chunk_mask` (the
  partial/finalize split lets callers insert a CP all-reduce between), plus
  :func:`nemo_rl.distributed.model_utils.group_all_reduce_sum` for the global
  valid-chunk denominator.
- Teacher-logit IPC: :func:`rebuild_teacher_full_logits_from_ipc`,
  :func:`assemble_teacher_logits_from_shards`,
  :func:`collect_overlapping_teacher_shards` reassemble full-vocab teacher
  logits from per-rank shards across heterogeneous TP/CP.
- Projection: :func:`parse_projection_file`, the
  :func:`get_sparse_projection_matrix` / :func:`get_topk_projection`
  process-local caches, :func:`slice_sparse_projection_rows`, and
  :func:`build_exact_token_map` (cached common/uncommon partition).
- :func:`alignment_from_flat_batch` rehydrates the flat ``alignment_*``
  data-dict keys into an :class:`AlignmentBatch`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch
from nemo_rl.distributed.model_utils import (
    ResolvedVocabParallelLogits,
    get_target_logprobs_from_vocab_parallel_logits,
    group_all_reduce_sum_with_grad,
    vocab_parallel_argmax,
)
from nemo_rl.models.dtensor.parallelize import to_local_if_dtensor


def alignment_from_flat_batch(data: Mapping[str, Any]) -> AlignmentBatch:
    """Rebuild :class:`AlignmentBatch` from the flat ``alignment_*`` keys.

    The field set is driven off :class:`AlignmentBatch` so the helper
    can't drift from the schema.
    """
    return AlignmentBatch(
        **{f.name: data[f"alignment_{f.name}"] for f in fields(AlignmentBatch)}
    )


class Fp32SparseMM(torch.autograd.Function):
    """FP32 ``M.t() @ dense`` (sparse-dense matmul) ignoring surrounding autocast.

    ``addmm_sparse_cuda`` has no BF16 kernel on either forward or backward.
    The worker wraps forward + loss + backward in ``autocast(BF16)``, so a
    plain ``with autocast(enabled=False):`` around the forward call is not
    enough — ``loss.backward()`` runs inside the outer autocast and the
    sparse-mm backward kernel is still dispatched as BF16. The
    ``custom_fwd(cast_inputs=torch.float32)`` / ``custom_bwd`` decorators
    are PyTorch's official escape: they force FP32 inputs on forward and
    run the backward as if autocast were disabled.

    autograd's builtin sparse-mm backward computes
    ``M @ grad_out``. The gradient w.r.t. the sparse argument isn't
    needed (the projection matrix is frozen), so it's returned as ``None``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx: Any, sparse_M: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        ctx.sparse_M = sparse_M
        return torch.sparse.mm(sparse_M.t(), dense)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[None, torch.Tensor]:
        sparse_M = ctx.sparse_M
        # out = sparse_M.t() @ dense, so d/d_dense = sparse_M @ grad_out.
        grad_dense = torch.sparse.mm(sparse_M, grad_out)
        return None, grad_dense


def chunk_log_prob_sums(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local bmm + bucket count, no division.

    Output is summable across CP; callers that need cross-rank chunks to
    aggregate correctly should ``group_all_reduce_sum_with_grad`` both tensors
    before :func:`chunk_average_finalize`. ``chunk_id == -1`` contributes to no
    bucket.
    """
    device = log_probs.device
    chunk_arange = torch.arange(max_chunks, device=device).view(1, 1, -1)
    chunk_mask = chunk_id.unsqueeze(-1) == chunk_arange
    chunk_mask_f = chunk_mask.transpose(1, 2).to(log_probs.dtype)
    chunk_sums = torch.bmm(chunk_mask_f, log_probs)  # [B, C, V]
    chunk_sizes = chunk_mask.sum(dim=1).float()  # [B, C]
    return chunk_sums, chunk_sizes


def chunk_average_finalize(
    chunk_sums: torch.Tensor,
    chunk_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide sums by sizes; ``eps`` guards empty buckets."""
    eps = 1e-10
    chunk_log_probs = chunk_sums / (chunk_sizes.unsqueeze(-1) + eps)
    return chunk_log_probs, chunk_sizes


def chunk_average_log_probs(
    log_probs: torch.Tensor,
    chunk_id: torch.Tensor,
    max_chunks: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``log_probs`` over chunks defined by ``chunk_id``.

    Builds a one-hot chunk mask from ``chunk_id`` (``-1`` = no chunk), then
    ``bmm``-aggregates and divides by chunk sizes. When ``cp_group`` has world
    > 1, the per-chunk sums are ``group_all_reduce_sum_with_grad``'d across CP
    ranks before the divide (mean is non-linear, so the reduce must precede it).

    Args:
        log_probs: ``[B, T, V]`` log-probabilities.
        chunk_id: ``[B, T]`` long tensor, values in ``[-1, max_chunks)``.
        max_chunks: number of chunk buckets.
        cp_group: context-parallel group for cross-rank chunk aggregation.

    Returns:
        chunk_log_probs: ``[B, max_chunks, V]`` averaged log-probs.
        chunk_sizes: ``[B, max_chunks]`` float tensor of bucket sizes.
    """
    chunk_sums, chunk_sizes = chunk_log_prob_sums(log_probs, chunk_id, max_chunks)
    if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
        chunk_sums = group_all_reduce_sum_with_grad(chunk_sums, cp_group)
        chunk_sizes = group_all_reduce_sum_with_grad(chunk_sizes, cp_group)
    return chunk_average_finalize(chunk_sums, chunk_sizes)


def slice_sparse_projection_rows(
    sparse_matrix: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Row-slice a sparse-COO projection ``[V_s, V_t]`` to ``[row_end-row_start, V_t]``.

    Filters COO indices in-place: keeps entries with row in ``[row_start, row_end)``
    and shifts the row index by ``-row_start``. Used by the TP-aware P-KL path
    where each rank owns a contiguous slab of the student vocab axis.
    """
    indices = sparse_matrix.indices()
    values = sparse_matrix.values()
    mask = (indices[0] >= row_start) & (indices[0] < row_end)
    local_indices = indices[:, mask].clone()
    local_indices[0] -= row_start
    local_values = values[mask]
    return torch.sparse_coo_tensor(
        local_indices,
        local_values,
        (row_end - row_start, sparse_matrix.size(1)),
        device=sparse_matrix.device,
        dtype=sparse_matrix.dtype,
    ).coalesce()


# ---------------------------------------------------------------------------
# TP/CP-aware loss primitives
#
# Each of these collapses to the plain single-rank torch op when the relevant
# process group has world size 1, so the cross-tokenizer loss body stays free
# of any ``tp_world > 1`` / rank / offset branching.
# ---------------------------------------------------------------------------
def project_student_to_teacher_vocab(
    student_probs: torch.Tensor,
    sparse_projection: torch.Tensor,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Project student vocab probs ``[B, T, V_s(/TP)]`` to teacher vocab ``[B, T, V_t]``.

    ``sparse_projection`` is the full ``[V_s, V_t]`` sparse-COO matrix. With
    ``tp_group`` world > 1 the student probs cover only this rank's ``V_s/TP``
    rows, so the matrix is row-sliced to that range, the sparse matmul produces a
    partial teacher-vocab sum, and a ``group_all_reduce_sum_with_grad`` over the
    TP group combines the partials into the full ``V_s`` contraction. Otherwise a
    single sparse matmul over the full matrix is used.
    """
    batch_size, seq_len, local_vocab_size = student_probs.shape
    flat = student_probs.reshape(batch_size * seq_len, local_vocab_size)
    tp_world = torch.distributed.get_world_size(tp_group) if tp_group is not None else 1
    if tp_world > 1:
        tp_rank = torch.distributed.get_rank(tp_group)
        full_student_vocab_size = sparse_projection.size(0)
        rows_per_rank = full_student_vocab_size // tp_world
        local_projection = slice_sparse_projection_rows(
            sparse_projection,
            row_start=tp_rank * rows_per_rank,
            row_end=(tp_rank + 1) * rows_per_rank,
        )
        projected_partial = Fp32SparseMM.apply(local_projection, flat.t()).t()
        projected = group_all_reduce_sum_with_grad(
            projected_partial.contiguous(), tp_group
        )
    else:
        # Fp32SparseMM internally computes M.t() @ dense; passing M (not M.t())
        # avoids a sparse ``.t()`` on a saved tensor in backward.
        projected = Fp32SparseMM.apply(sparse_projection, flat.t()).t()
    teacher_vocab_size = projected.shape[-1]
    return projected.reshape(batch_size, seq_len, teacher_vocab_size)


def select_teacher_topk_indices(
    teacher_logits: torch.Tensor,
    k: int,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Sorted global top-``k`` teacher-vocab ids by max importance over the microbatch.

    Importance is the per-vocab max over flattened ``(B*T)`` teacher logits. With
    ``cp_group`` world > 1 the sequence is CP-sharded, so the local max only sees
    this rank's slice; an ``all_reduce(MAX)`` makes every rank pick the same
    subset. No gradient.
    """
    vocab_size = teacher_logits.shape[-1]
    with torch.no_grad():
        # reshape (not view): a preceding next-token shift can leave the teacher
        # logits non-contiguous.
        teacher_flat = teacher_logits.reshape(-1, vocab_size)
        importance = teacher_flat.max(dim=0).values
        if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
            torch.distributed.all_reduce(
                importance, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )
        top_indices = torch.topk(importance, k=k, dim=-1).indices
        return top_indices.sort().values


@dataclass(frozen=True)
class LocalizedAlignment:
    """CP-localized alignment tensors consumed by the loss reductions.

    For a cross-tokenizer teacher every field is populated (chunk-averaged
    projection KL / gold path and teacher-local scoring). For a same-tokenizer
    teacher only ``sample_mask`` is populated; shared student tensors stay in
    the top-level loss contract.
    """

    sample_mask: torch.Tensor
    student_chunk_id: Optional[torch.Tensor] = None
    teacher_chunk_id: Optional[torch.Tensor] = None
    pair_valid: Optional[torch.Tensor] = None
    pair_is_correct: Optional[torch.Tensor] = None
    teacher_token_mask: Optional[torch.Tensor] = None
    teacher_next_token_ids: Optional[torch.Tensor] = None
    teacher_next_token_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        """Validate the self-contained same-vocab or cross-vocab shape contract."""
        if self.sample_mask.ndim != 1:
            raise ValueError(
                f"sample_mask must be one-dimensional, got {self.sample_mask.shape}."
            )
        alignment_fields = {
            "student_chunk_id": self.student_chunk_id,
            "teacher_chunk_id": self.teacher_chunk_id,
            "pair_valid": self.pair_valid,
            "pair_is_correct": self.pair_is_correct,
            "teacher_token_mask": self.teacher_token_mask,
            "teacher_next_token_ids": self.teacher_next_token_ids,
            "teacher_next_token_mask": self.teacher_next_token_mask,
        }
        present = [
            name for name, value in alignment_fields.items() if value is not None
        ]
        if not present:
            return
        missing = [name for name, value in alignment_fields.items() if value is None]
        if missing:
            raise ValueError(
                "Cross-tokenizer alignment fields must be populated together; "
                f"missing {missing}."
            )

        assert self.student_chunk_id is not None
        assert self.teacher_chunk_id is not None
        assert self.pair_valid is not None
        assert self.pair_is_correct is not None
        assert self.teacher_token_mask is not None
        assert self.teacher_next_token_ids is not None
        assert self.teacher_next_token_mask is not None
        batch_size = self.sample_mask.shape[0]
        tensors = {
            "student_chunk_id": self.student_chunk_id,
            "teacher_chunk_id": self.teacher_chunk_id,
            "pair_valid": self.pair_valid,
            "pair_is_correct": self.pair_is_correct,
            "teacher_token_mask": self.teacher_token_mask,
            "teacher_next_token_ids": self.teacher_next_token_ids,
            "teacher_next_token_mask": self.teacher_next_token_mask,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 2 or tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} must have shape [B, S] with B={batch_size}, got "
                    f"{tensor.shape}."
                )
        if self.pair_valid.shape != self.pair_is_correct.shape:
            raise ValueError(
                "pair_valid and pair_is_correct must have identical shapes, got "
                f"{self.pair_valid.shape} and {self.pair_is_correct.shape}."
            )
        teacher_shape = self.teacher_chunk_id.shape
        for name, tensor in (
            ("teacher_token_mask", self.teacher_token_mask),
            ("teacher_next_token_ids", self.teacher_next_token_ids),
            ("teacher_next_token_mask", self.teacher_next_token_mask),
        ):
            if tensor.shape != teacher_shape:
                raise ValueError(
                    f"{name} must match teacher_chunk_id shape {teacher_shape}, "
                    f"got {tensor.shape}."
                )


def aligned_next_token_accuracy(
    logits: torch.Tensor,
    *,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup],
    cp_group: Optional[torch.distributed.ProcessGroup],
    dp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Compute top-1 accuracy from already aligned local predictor targets."""
    target_ids = to_local_if_dtensor(target_ids).to(logits.device)
    target_mask = to_local_if_dtensor(target_mask).to(logits.device)
    sample_mask = to_local_if_dtensor(sample_mask).to(logits.device)
    expected = tuple(logits.shape[:2])
    if tuple(target_ids.shape) != expected or tuple(target_mask.shape) != expected:
        raise ValueError(
            "Aligned accuracy targets must match the logits [B, S] prefix, got "
            f"logits={logits.shape}, ids={target_ids.shape}, mask={target_mask.shape}."
        )
    if target_ids.dtype != torch.long:
        raise TypeError(
            f"target_ids must have dtype torch.long, got {target_ids.dtype}."
        )
    if sample_mask.ndim != 1 or sample_mask.shape[0] != logits.shape[0]:
        raise ValueError(
            f"sample_mask must have shape [{logits.shape[0]}], got {sample_mask.shape}."
        )

    with torch.no_grad():
        argmax = vocab_parallel_argmax(logits, tp_group=tp_group)
        mask = target_mask.float() * sample_mask.unsqueeze(-1).float()
        stats = torch.stack([((argmax == target_ids).float() * mask).sum(), mask.sum()])
        for group in (cp_group, dp_group):
            if group is not None:
                torch.distributed.all_reduce(stats, group=group)
        return stats[0] / stats[1].clamp(min=1.0)


def collect_overlapping_teacher_shards(
    teacher_shards: list[dict[str, Any]],
    student_cp_rank: int,
    student_cp_size: int,
    full_seq_len: int,
) -> list[tuple[dict[str, Any], slice, slice, slice, slice]]:
    """Plan ``(src_seq, src_vocab, dest_seq, dest_vocab)`` slices per teacher shard.

    Dest is ``[T_t/CP_s, V_t]`` (vocab fully reassembled, seq is this
    student CP rank's range). Shards with no seq overlap are skipped.
    """
    student_seq_start = student_cp_rank * full_seq_len // student_cp_size
    student_seq_end = (student_cp_rank + 1) * full_seq_len // student_cp_size

    matches: list[tuple[dict[str, Any], slice, slice, slice, slice]] = []
    for handle in teacher_shards:
        teacher_vocab_start = int(handle["vocab_start_index"])
        teacher_vocab_end = int(handle["vocab_end_index"])
        teacher_seq_start = int(handle["global_seq_start"])
        teacher_seq_end = teacher_seq_start + int(handle["actual_shape"][0])

        overlap_seq_start = max(student_seq_start, teacher_seq_start)
        overlap_seq_end = min(student_seq_end, teacher_seq_end)
        if overlap_seq_end <= overlap_seq_start:
            continue

        src_seq = slice(
            overlap_seq_start - teacher_seq_start,
            overlap_seq_end - teacher_seq_start,
        )
        src_vocab = slice(0, teacher_vocab_end - teacher_vocab_start)
        dest_seq = slice(
            overlap_seq_start - student_seq_start,
            overlap_seq_end - student_seq_start,
        )
        dest_vocab = slice(teacher_vocab_start, teacher_vocab_end)
        matches.append((handle, src_seq, src_vocab, dest_seq, dest_vocab))
    return matches


def _teacher_ipc_window_coordinates(
    per_sample_entries: list[dict[str, Any]],
    *,
    student_cp_rank: int,
    student_cp_size: int,
) -> tuple[int, int, int]:
    """Resolve one student's contiguous teacher window from validated IPC metadata."""
    if not per_sample_entries:
        raise ValueError("Teacher IPC entries must be non-empty.")
    if student_cp_size <= 0 or not 0 <= student_cp_rank < student_cp_size:
        raise ValueError(
            f"Invalid student CP topology: rank={student_cp_rank}, size={student_cp_size}."
        )

    first_shards = per_sample_entries[0].get("teacher_shards")
    if not first_shards:
        raise ValueError("Every teacher IPC sample must contain teacher_shards.")
    full_seq_len = int(first_shards[0]["full_seq_len"])
    full_vocab_size = int(first_shards[0]["full_vocab_size"])
    if full_seq_len <= 0 or full_vocab_size <= 0:
        raise ValueError(
            f"Invalid teacher IPC shape: T={full_seq_len}, V={full_vocab_size}."
        )
    if full_seq_len % student_cp_size != 0:
        raise ValueError(
            "Teacher sequence length must be divisible by the student CP size, "
            f"got T={full_seq_len}, CP={student_cp_size}."
        )
    local_seq_len = full_seq_len // student_cp_size
    global_seq_start = student_cp_rank * local_seq_len
    return global_seq_start, local_seq_len, full_seq_len


def _resolve_teacher_ipc_window(
    per_sample_entries: list[dict[str, Any]],
    *,
    student_cp_rank: int,
    student_cp_size: int,
) -> tuple[int, int, int]:
    """Resolve and fully validate one student's contiguous teacher IPC window."""
    global_seq_start, local_seq_len, full_seq_len = _teacher_ipc_window_coordinates(
        per_sample_entries,
        student_cp_rank=student_cp_rank,
        student_cp_size=student_cp_size,
    )
    first_shards = per_sample_entries[0]["teacher_shards"]
    full_vocab_size = int(first_shards[0]["full_vocab_size"])

    for sample_idx, entry in enumerate(per_sample_entries):
        shards = entry.get("teacher_shards")
        if not shards:
            raise ValueError(f"Teacher IPC sample {sample_idx} has no shards.")
        for shard in shards:
            if (
                int(shard["full_seq_len"]) != full_seq_len
                or int(shard["full_vocab_size"]) != full_vocab_size
            ):
                raise ValueError(
                    f"Teacher IPC sample {sample_idx} has inconsistent global shape."
                )
        matches = collect_overlapping_teacher_shards(
            shards,
            student_cp_rank=student_cp_rank,
            student_cp_size=student_cp_size,
            full_seq_len=full_seq_len,
        )
        rectangles: list[tuple[int, int, int, int]] = []
        covered_area = 0
        for _, _, _, dest_seq, dest_vocab in matches:
            seq_start = int(dest_seq.start or 0)
            seq_end = int(dest_seq.stop or 0)
            vocab_start = int(dest_vocab.start or 0)
            vocab_end = int(dest_vocab.stop or 0)
            rectangles.append((seq_start, seq_end, vocab_start, vocab_end))
            covered_area += (seq_end - seq_start) * (vocab_end - vocab_start)
        for index, lhs in enumerate(rectangles):
            for rhs in rectangles[index + 1 :]:
                if max(lhs[0], rhs[0]) < min(lhs[1], rhs[1]) and max(
                    lhs[2], rhs[2]
                ) < min(lhs[3], rhs[3]):
                    raise ValueError(
                        f"Teacher IPC sample {sample_idx} has overlapping window "
                        f"coverage: {lhs} and {rhs}."
                    )
        expected_area = local_seq_len * full_vocab_size
        if covered_area != expected_area:
            raise ValueError(
                f"Teacher IPC sample {sample_idx} covers area {covered_area}, "
                f"expected {expected_area} for the requested student window."
            )
    return global_seq_start, local_seq_len, full_seq_len


def validate_xtoken_window_contract(
    data: Mapping[str, Any],
    *,
    projection_matrix_paths: list[Optional[str]],
    student_seq_len: int,
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
) -> None:
    """Validate equal student and teacher windows before student forward."""
    if not projection_matrix_paths:
        raise ValueError("x-token requires at least one teacher.")
    cp_rank = (
        torch.distributed.get_rank(context_parallel_group)
        if context_parallel_group is not None
        else 0
    )
    cp_size = (
        torch.distributed.get_world_size(context_parallel_group)
        if context_parallel_group is not None
        else 1
    )
    if student_seq_len <= 0 or student_seq_len % cp_size != 0:
        raise ValueError(
            "x-token requires the canonical student sequence length to be "
            f"positive and divisible by CP size, got T={student_seq_len}, CP={cp_size}."
        )
    student_batch_size = int(data["input_ids"].shape[0])
    for i, projection_path in enumerate(projection_matrix_paths):
        key = f"teacher_{i}_full_logits_ipc"
        if key not in data:
            raise KeyError(f"Missing x-token teacher IPC field {key!r}.")
        if len(data[key]) != student_batch_size:
            raise ValueError(
                f"Teacher {i} IPC entries must match student batch size "
                f"{student_batch_size}, got {len(data[key])}."
            )
        _, _, teacher_full_seq_len = _resolve_teacher_ipc_window(
            data[key], student_cp_rank=cp_rank, student_cp_size=cp_size
        )
        if projection_path is None:
            if teacher_full_seq_len != student_seq_len:
                raise ValueError(
                    f"Same-vocabulary teacher {i} must use the student sequence "
                    f"length {student_seq_len}, got {teacher_full_seq_len}."
                )
            continue
        required = (
            f"teacher_{i}_input_ids",
            f"teacher_{i}_token_mask",
            f"alignment_{i}_student_chunk_id",
            f"alignment_{i}_teacher_chunk_id",
            f"alignment_{i}_pair_valid",
            f"alignment_{i}_pair_is_correct",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Teacher {i} is missing x-token fields: {missing}.")
        teacher_ids = to_local_if_dtensor(data[required[0]])
        teacher_mask = to_local_if_dtensor(data[required[1]])
        teacher_shape = (student_batch_size, teacher_full_seq_len)
        if (
            tuple(teacher_ids.shape) != teacher_shape
            or tuple(teacher_mask.shape) != teacher_shape
        ):
            raise ValueError(
                f"Teacher {i} IDs/mask must have shape {teacher_shape}, got "
                f"{teacher_ids.shape} and {teacher_mask.shape}."
            )
        if teacher_ids.dtype != torch.long:
            raise TypeError(
                f"Teacher {i} input IDs must have dtype torch.long, got "
                f"{teacher_ids.dtype}."
            )
        full_vocab_size = int(data[key][0]["teacher_shards"][0]["full_vocab_size"])
        if teacher_ids.numel() > 0:
            min_id = int(teacher_ids.min().item())
            max_id = int(teacher_ids.max().item())
            if min_id < 0 or max_id >= full_vocab_size:
                raise ValueError(
                    f"Teacher {i} token IDs are outside [0, {full_vocab_size}): "
                    f"min={min_id}, max={max_id}."
                )
        student_chunks = to_local_if_dtensor(data[required[2]])
        teacher_chunks = to_local_if_dtensor(data[required[3]])
        if tuple(student_chunks.shape) != (student_batch_size, student_seq_len):
            raise ValueError(
                f"Teacher {i} student chunk IDs must have shape "
                f"{(student_batch_size, student_seq_len)}, got {student_chunks.shape}."
            )
        if tuple(teacher_chunks.shape) != teacher_shape:
            raise ValueError(
                f"Teacher {i} teacher chunk IDs must have shape {teacher_shape}, "
                f"got {teacher_chunks.shape}."
            )
        pair_valid = to_local_if_dtensor(data[required[4]])
        pair_correct = to_local_if_dtensor(data[required[5]])
        if (
            pair_valid.ndim != 2
            or pair_valid.shape != pair_correct.shape
            or pair_valid.shape[0] != student_batch_size
        ):
            raise ValueError(
                f"Teacher {i} pair masks must have matching [B, C] shapes, got "
                f"{pair_valid.shape} and {pair_correct.shape}."
            )
        if pair_valid.dtype != torch.bool or pair_correct.dtype != torch.bool:
            raise TypeError(
                f"Teacher {i} pair masks must have dtype torch.bool, got "
                f"{pair_valid.dtype} and {pair_correct.dtype}."
            )
        if student_chunks.dtype != torch.long or teacher_chunks.dtype != torch.long:
            raise TypeError(
                f"Teacher {i} chunk IDs must have dtype torch.long, got "
                f"{student_chunks.dtype} and {teacher_chunks.dtype}."
            )
        max_chunks = int(pair_valid.shape[1])
        for name, chunks in (
            ("student", student_chunks),
            ("teacher", teacher_chunks),
        ):
            if chunks.numel() == 0:
                continue
            min_chunk = int(chunks.min().item())
            max_chunk = int(chunks.max().item())
            if min_chunk < -1 or max_chunk >= max_chunks:
                raise ValueError(
                    f"Teacher {i} {name} chunk IDs must lie in "
                    f"[-1, {max_chunks}), got min={min_chunk}, max={max_chunk}."
                )


def assemble_teacher_logits_from_shards(
    teacher_shards: list[dict[str, Any]],
    student_cp_rank: int,
    student_cp_size: int,
    device: int,
) -> torch.Tensor:
    """P2P-IPC-read overlapping teacher shards into a ``[T_t/CP_s, V_t]`` dest.

    ``device`` is a CUDA device index (matches
    :func:`rebuild_cuda_tensor_from_ipc`'s ``device_id`` signature).
    """
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    if not teacher_shards:
        raise ValueError("teacher_shards must be non-empty")
    full_seq_len = int(teacher_shards[0]["full_seq_len"])
    full_vocab_size = int(teacher_shards[0]["full_vocab_size"])
    # CP seq-padding guarantees this; assert it so the contiguous-window math
    # below (and the `dest` size) can't silently go out of bounds if a caller
    # ever passes an unpadded length.
    assert full_seq_len % student_cp_size == 0, (
        f"full_seq_len={full_seq_len} not divisible by student_cp_size={student_cp_size}"
    )
    local_seq_len = full_seq_len // student_cp_size

    dest = torch.zeros(
        (local_seq_len, full_vocab_size),
        dtype=torch.float32,
        device=device,
    )
    matches = collect_overlapping_teacher_shards(
        teacher_shards,
        student_cp_rank=student_cp_rank,
        student_cp_size=student_cp_size,
        full_seq_len=full_seq_len,
    )
    for handle, src_seq, src_vocab, dest_seq, dest_vocab in matches:
        # Producer's IPC payload is the full contiguous storage
        # [N_microbatches, B_mb, T_t_local, V_t_local]; index the slot
        # then the sample row, then apply the seq/vocab overlap slices.
        src_full = rebuild_cuda_tensor_from_ipc(handle["payload_ipc"], device).detach()
        buf_idx = int(handle["buf_idx"])
        sample_idx = int(handle["sample_index_in_buf"])
        local_seq_t, local_vocab_t = handle["actual_shape"]
        src = src_full[buf_idx, sample_idx, :local_seq_t, :local_vocab_t]
        dest[dest_seq, dest_vocab] = src[src_seq, src_vocab].to(torch.float32)
    return dest


def _try_zero_copy_teacher_logits(
    per_sample_entries: list[dict[str, Any]],
    *,
    student_cp_rank: int,
    student_cp_size: int,
    device: int,
) -> Optional[torch.Tensor]:
    """Zero-copy ``[B, T_t/CP_s, V_t]`` view of the teacher logits, or None.

    Returns a view into the producer's IPC storage only when reassembly is
    unnecessary: every sample's seq range is covered by a single full-vocab
    teacher shard (i.e. teacher ``tp_size == 1`` and ``teacher_cp == student_cp``
    or ``teacher_cp == 1``), and the microbatch's samples are a contiguous slab
    (same payload + ``buf_idx``, sample rows ``0..B-1``) in one storage slot.
    Otherwise returns None and the caller falls back to assemble + stack.
    """
    if not per_sample_entries:
        return None
    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    first_shards = per_sample_entries[0]["teacher_shards"]
    if not first_shards:
        return None
    full_seq_len = int(first_shards[0]["full_seq_len"])
    full_vocab_size = int(first_shards[0]["full_vocab_size"])
    student_seq_start = student_cp_rank * full_seq_len // student_cp_size
    student_seq_end = (student_cp_rank + 1) * full_seq_len // student_cp_size

    # Exactly one full-vocab shard must cover this student rank's seq range.
    chosen: list[dict[str, Any]] = []
    for entry in per_sample_entries:
        covering = [
            h
            for h in entry["teacher_shards"]
            if int(h["vocab_start_index"]) == 0
            and int(h["vocab_end_index"]) == full_vocab_size
            and int(h["global_seq_start"]) <= student_seq_start
            and int(h["global_seq_start"]) + int(h["actual_shape"][0])
            >= student_seq_end
        ]
        if len(covering) != 1:
            return None
        chosen.append(covering[0])

    # All samples must form a contiguous slab in one storage slot.
    h0 = chosen[0]
    payload = h0["payload_ipc"]
    buf_idx = int(h0["buf_idx"])
    teacher_seq_start = int(h0["global_seq_start"])
    for i, h in enumerate(chosen):
        if (
            h["payload_ipc"] != payload
            or int(h["buf_idx"]) != buf_idx
            or int(h["sample_index_in_buf"]) != i
            or int(h["global_seq_start"]) != teacher_seq_start
        ):
            return None

    src_full = rebuild_cuda_tensor_from_ipc(payload, device).detach()
    seq_lo = student_seq_start - teacher_seq_start
    seq_hi = student_seq_end - teacher_seq_start
    return src_full[buf_idx, : len(chosen), seq_lo:seq_hi, :full_vocab_size]


def rebuild_teacher_full_logits_from_ipc(
    per_sample_entries: list[dict[str, Any]],
    cp_group: Optional[torch.distributed.ProcessGroup],
    device: int,
) -> torch.Tensor:
    """Rebuild ``[B, T_t/CP_s, V_t]`` teacher logits for this student rank.

    Fast path (zero-copy view via :func:`_try_zero_copy_teacher_logits`): when the
    teacher is not vocab-sharded and each sample's seq range is covered by a
    single shard, return a view into the IPC storage. Otherwise reassemble each
    sample from its overlapping shards and stack.
    """
    student_cp_rank = (
        torch.distributed.get_rank(cp_group) if cp_group is not None else 0
    )
    student_cp_size = (
        torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
    )
    _resolve_teacher_ipc_window(
        per_sample_entries,
        student_cp_rank=student_cp_rank,
        student_cp_size=student_cp_size,
    )

    # Bypass: when the teacher layout lines up with this student rank (no
    # vocab sharding, seq covered by one shard), skip reassembly and return a
    # zero-copy view of the IPC storage. Returns None when reassembly is needed.
    view = _try_zero_copy_teacher_logits(
        per_sample_entries,
        student_cp_rank=student_cp_rank,
        student_cp_size=student_cp_size,
        device=device,
    )
    if view is not None:
        return view

    rebuilt = [
        assemble_teacher_logits_from_shards(
            entry["teacher_shards"],
            student_cp_rank=student_cp_rank,
            student_cp_size=student_cp_size,
            device=device,
        )
        for entry in per_sample_entries
    ]
    return torch.stack(rebuilt, dim=0)


def valid_chunk_mask(
    s_sizes: torch.Tensor,
    t_sizes: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    """Per-chunk validity gate: both sides non-empty and pair is valid."""
    return (s_sizes > 0) & (t_sizes > 0) & pair_valid


def parse_projection_file(
    path: Union[str, os.PathLike],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Parse a projection-matrix file into COO components.

    Detects either the dense top-k format (``dict["indices"]`` /
    ``dict["likelihoods"]``) or the sparse multi-token format
    (``dict[(student_id, teacher_id)] -> count``) and converts both to
    a uniform COO representation.

    The function does **not** apply any sizing or validity policy: the
    ``-1`` sentinel used by ``_exact_map_remapped`` projection files is
    preserved in the returned ``indices``, and the inferred vocab sizes
    are derived from the file alone (caller may override them upward
    against tokenizer / config knowledge). This keeps a single parser
    while letting :mod:`token_aligner` and the loss fn keep their own
    clipping rules.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.

    Returns:
        indices: ``LongTensor[2, nnz]`` — ``(student_idx, teacher_idx)``.
        values:  ``FloatTensor[nnz]``.
        v_student_inferred: ``int`` — dense format: row count; sparse
            format: ``max(student_idx) + 1``.
        v_teacher_inferred: ``int`` — ``max(positive teacher_idx) + 1``
            (``0`` if no positive entries exist).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file is not in a recognized format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection matrix file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "indices" in data and "likelihoods" in data:
        # Dense top-k format: indices [V_s, top_k] holds teacher token ids;
        # likelihoods [V_s, top_k] holds the projection weights. Unfold to
        # COO so downstream code uses a uniform sparse-matmul path.
        top_indices: torch.Tensor = data["indices"].long()
        top_likelihoods: torch.Tensor = data["likelihoods"].float()
        if top_indices.shape != top_likelihoods.shape:
            raise ValueError(
                f"indices/likelihoods shape mismatch in {path}: "
                f"{top_indices.shape} vs {top_likelihoods.shape}"
            )
        v_student, top_k = top_indices.shape
        student_idx = torch.arange(v_student).unsqueeze(1).expand(-1, top_k).reshape(-1)
        teacher_idx = top_indices.reshape(-1)
        values = top_likelihoods.reshape(-1)
        indices = torch.stack([student_idx, teacher_idx], dim=0)
        positive = teacher_idx[teacher_idx >= 0]
        v_teacher = int(positive.max().item()) + 1 if positive.numel() > 0 else 0
        return indices, values, int(v_student), v_teacher

    if isinstance(data, dict) and all(
        isinstance(k, tuple) and len(k) == 2 for k in data.keys()
    ):
        # Sparse multi-token format: dict[(student_id, teacher_id)] -> count.
        keys = list(data.keys())
        values_list = list(data.values())
        student_idx = torch.tensor([k[0] for k in keys], dtype=torch.long)
        teacher_idx = torch.tensor([k[1] for k in keys], dtype=torch.long)
        indices = torch.stack([student_idx, teacher_idx], dim=0)
        values = torch.tensor(values_list, dtype=torch.float32)
        v_student = int(student_idx.max().item()) + 1 if student_idx.numel() > 0 else 0
        v_teacher = int(teacher_idx.max().item()) + 1 if teacher_idx.numel() > 0 else 0
        return indices, values, v_student, v_teacher

    raise ValueError(
        f"Unrecognized projection matrix format at {path}; expected dict "
        f"with 'indices'/'likelihoods' tensors or "
        f"dict[(student_id, teacher_id)] -> count."
    )


# Process-local projection-matrix caches. Each Ray worker / dataloader
# process has its own Python interpreter, so these dicts are effectively
# worker-local: a cache miss on one worker doesn't fill caches on other
# workers, and the driver process — which never enters a forward / loss
# path — never populates them.
#
# Keyed by ``(path, device, student_vocab_size, teacher_vocab_size)`` for
# the sparse cache because the sparse-COO shape's ``V_s`` and ``V_t`` are
# both sized from the configured vocab sizes; same path with a different
# size would build a different tensor. The top-k cache key is
# ``(path, device)`` — the raw top-k arrays don't depend on a vocab-size
# knob.
_SPARSE_PROJECTION_CACHE: dict[Tuple[str, torch.device, int, int], torch.Tensor] = {}
_TOPK_PROJECTION_CACHE: dict[
    Tuple[str, torch.device], Tuple[torch.Tensor, torch.Tensor]
] = {}


def get_sparse_projection_matrix(
    path: Union[str, os.PathLike],
    device: torch.device,
    *,
    student_vocab_size: int,
    teacher_vocab_size: int,
) -> torch.Tensor:
    """Return the sparse-COO projection matrix on ``device`` (cached).

    On a cache miss, parses the file via :func:`parse_projection_file`,
    drops ``-1`` teacher sentinels (illegal in sparse-COO), sizes
    ``V_s = max(student_vocab_size, max_observed_student_idx + 1)`` and
    ``V_t = max(teacher_vocab_size, max_observed_teacher_idx + 1)``, and
    builds a coalesced ``torch.sparse_coo_tensor`` on ``device``.
    Subsequent calls with the same
    ``(path, device, student_vocab_size, teacher_vocab_size)`` return the
    cached tensor — no disk I/O, no re-materialization.

    Both vocab sizes are keyword-only to prevent a positional swap (two
    same-magnitude ints, no error if confused).

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the sparse tensor must live on.
        student_vocab_size: Minimum width of the student-side axis.
        teacher_vocab_size: Minimum width of the teacher-side axis.

    Returns:
        ``torch.sparse_coo_tensor`` of shape ``(V_s, V_t)``, coalesced,
        ``dtype=float32``.
    """
    key = (
        str(path),
        device,
        int(student_vocab_size),
        int(teacher_vocab_size),
    )
    cached = _SPARSE_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    indices, values, _v_student, _ = parse_projection_file(path)
    # `_exact_map_remapped` projection files use -1 as a padding
    # sentinel for student rows that have fewer than top_k teacher
    # mappings. A negative column index is illegal in a sparse tensor
    # and causes CUDA illegal-memory-access in sparse.mm (forward and
    # backward). We drop those entries entirely.
    keep = indices[1] >= 0
    indices = indices[:, keep]
    values = values[keep]
    # Size both axes from the configured tokenizer vocabs, not from the
    # highest ids observed in the projection file. The sparse format
    # only stores entries for (student_id, teacher_id) pairs that
    # appeared during projection prep, so the highest valid vocab ids
    # may be absent. Sizing V_s from `max(observed student_id)+1` would
    # then make V_s < logits.shape[-1] and silently break the sparse
    # matmul; the symmetric concern on V_t lets the P-KL global top-k
    # gather go out of bounds. We clamp up against the projection's
    # observed max as a defensive fallback in case the file happens to
    # cover ids beyond the configured size.
    projection_max_student = (
        int(indices[0].max().item()) + 1 if indices.numel() > 0 else 0
    )
    projection_max_teacher = (
        int(indices[1].max().item()) + 1 if indices.numel() > 0 else 0
    )
    v_student = max(int(student_vocab_size), projection_max_student)
    v_teacher = max(int(teacher_vocab_size), projection_max_teacher)

    sparse = torch.sparse_coo_tensor(
        indices,
        values,
        (v_student, v_teacher),
        device=device,
        dtype=torch.float32,
    ).coalesce()
    _SPARSE_PROJECTION_CACHE[key] = sparse
    return sparse


def get_topk_projection(
    path: Union[str, os.PathLike],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the dense top-k ``(indices, likelihoods)`` projection on ``device`` (cached).

    Used by the gold-loss exact-map builder, which needs the per-row
    top-k weights — the sparse ``dict[(s, t)] -> count`` projection
    format doesn't carry those, so this loader rejects it.

    Args:
        path: Path to a ``torch.save``d projection-matrix file.
        device: Device the returned tensors must live on.

    Returns:
        ``(indices, likelihoods)`` — ``LongTensor[V_s, top_k]`` and
        ``FloatTensor[V_s, top_k]`` on ``device``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file is not in the dense top-k format.
    """
    key = (str(path), device)
    cached = _TOPK_PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached

    if not os.path.exists(path):
        raise FileNotFoundError(f"Projection matrix file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "indices" in data and "likelihoods" in data):
        raise ValueError(
            f"gold_loss requires the dense projection-matrix format "
            f"(dict with 'indices' and 'likelihoods' tensors). File "
            f"{path} uses an unsupported format."
        )
    indices = data["indices"].long().to(device)
    likelihoods = data["likelihoods"].float().to(device)
    result = (indices, likelihoods)
    _TOPK_PROJECTION_CACHE[key] = result
    return result


# Process-local cache. Keyed by every input that affects the partition:
# the same file with a different ``xtoken_loss`` or ``teacher_vocab_size``
# would yield a different partition. Lives alongside
# ``_TOPK_PROJECTION_CACHE`` so the gold-loss build is amortized to one
# pass per (path, device, knob) on each worker.
_EXACT_TOKEN_MAP_CACHE: dict[
    Tuple[str, torch.device, bool, int], Dict[str, torch.Tensor]
] = {}


def build_exact_token_map(
    path: Union[str, os.PathLike],
    device: torch.device,
    *,
    xtoken_loss: bool,
    teacher_vocab_size: int,
) -> Dict[str, torch.Tensor]:
    """Build the common/uncommon vocab partition for the gold path (cached).

    Reads the dense projection arrays via :func:`get_topk_projection`, sorts each
    student row's projection weights descending, then picks an exact-token
    map per the ``xtoken_loss`` flag:

    - ``xtoken_loss=False`` (strict): ``has_exact_map = (sorted_values[:, 0] == 1.0) & (projection_indices[:, 1] == -1)``.
      On collision (multiple students mapping to the same teacher id),
      the earliest (lowest) student index wins.
    - ``xtoken_loss=True`` (relaxed): ``has_exact_map = sorted_values[:, 0] >= 0.6``.
      On collision, the student with the highest first-projection
      weight wins; ties are broken by lowest student index.

    Both branches are vectorized via ``scatter_reduce`` so the build is
    O(V_s) and happens once per ``(path, device, xtoken_loss,
    teacher_vocab_size)`` for the run.

    Args:
        path: Path to a ``torch.save``d projection-matrix file (dense
            top-k format).
        device: Device the returned tensors must live on.
        xtoken_loss: Selects strict vs relaxed exact-map rule (see above).
        teacher_vocab_size: Width of the teacher-side vocab axis. The
            partition is bounded by this — teacher ids outside the range
            are dropped.

    Returns:
        Dict with keys ``common_student``, ``common_teacher`` (paired),
        ``uncommon_student``, ``uncommon_teacher`` (each independently
        sorted). All ``[long]`` tensors on ``device``.
    """
    key = (str(path), device, bool(xtoken_loss), int(teacher_vocab_size))
    cached = _EXACT_TOKEN_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    indices, likelihoods = get_topk_projection(path, device)
    v_student = indices.shape[0]
    v_teacher = int(teacher_vocab_size)

    sorted_values, sorted_in_topk = torch.sort(likelihoods, dim=-1, descending=True)
    if xtoken_loss:
        has_exact_map = sorted_values[:, 0] >= 0.6
    else:
        # Strict: exactly one top-k entry with weight 1.0, no second
        # mapping. `indices[:, 1] == -1` is the sentinel used by the
        # `_exact_map_remapped` projection files for "no second
        # mapping".
        has_exact_map = (sorted_values[:, 0] == 1.0) & (indices[:, 1] == -1)

    # Gather (s_idx, t_idx, prob) for each exact-map candidate.
    s_candidates = torch.where(has_exact_map)[0]
    if s_candidates.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        result = {
            "common_student": empty,
            "common_teacher": empty,
            "uncommon_student": torch.arange(v_student, device=device),
            "uncommon_teacher": torch.arange(v_teacher, device=device),
        }
        _EXACT_TOKEN_MAP_CACHE[key] = result
        return result

    t_candidates = indices[s_candidates, sorted_in_topk[s_candidates, 0]]
    prob_candidates = sorted_values[s_candidates, 0]

    in_bounds = (t_candidates >= 0) & (t_candidates < v_teacher)
    s_vec = s_candidates[in_bounds]
    t_vec = t_candidates[in_bounds]
    prob_vec = prob_candidates[in_bounds]

    # Strict mode: any candidate is eligible (first one wins).
    # Relaxed mode: only candidates whose prob ties the per-teacher max.
    if xtoken_loss:
        max_prob_per_t = torch.full(
            (v_teacher,),
            float("-inf"),
            device=device,
            dtype=prob_vec.dtype,
        )
        max_prob_per_t.scatter_reduce_(
            0, t_vec, prob_vec, reduce="amax", include_self=True
        )
        eligible = prob_vec >= max_prob_per_t[t_vec]
    else:
        eligible = torch.ones_like(t_vec, dtype=torch.bool)

    # For each teacher id, pick the smallest student index among the
    # eligible candidates. Sentinel = v_student so non-eligible rows
    # lose the amin reduction.
    sentinel = torch.tensor(v_student, dtype=s_vec.dtype, device=device)
    eligible_s = torch.where(eligible, s_vec, sentinel.expand_as(s_vec))
    min_s_per_t = torch.full((v_teacher,), v_student, device=device, dtype=s_vec.dtype)
    min_s_per_t.scatter_reduce_(0, t_vec, eligible_s, reduce="amin", include_self=True)
    winner_mask = eligible & (s_vec == min_s_per_t[t_vec])

    common_student = s_vec[winner_mask]
    common_teacher = t_vec[winner_mask]
    # Sort by student index so the paired arrays match.
    sort_perm = torch.argsort(common_student)
    common_student = common_student[sort_perm]
    common_teacher = common_teacher[sort_perm]

    common_s_mask = torch.zeros(v_student, dtype=torch.bool, device=device)
    common_s_mask[common_student] = True
    common_t_mask = torch.zeros(v_teacher, dtype=torch.bool, device=device)
    common_t_mask[common_teacher] = True
    uncommon_student = (~common_s_mask).nonzero(as_tuple=True)[0]
    uncommon_teacher = (~common_t_mask).nonzero(as_tuple=True)[0]

    result = {
        "common_student": common_student,
        "common_teacher": common_teacher,
        "uncommon_student": uncommon_student,
        "uncommon_teacher": uncommon_teacher,
    }
    _EXACT_TOKEN_MAP_CACHE[key] = result
    return result


def _prepare_xtoken_teacher_window_loss_inputs(
    data: Mapping[str, Any],
    *,
    student_seq_start: int,
    student_seq_len: int,
    student_cp_rank: int,
    student_cp_size: int,
    projection_matrix_paths: list[Optional[str]],
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
    device: torch.device,
) -> tuple[Dict[int, torch.Tensor], Dict[int, LocalizedAlignment]]:
    """Rebuild teacher logits and localize canonical x-token metadata."""
    if device.type != "cuda":
        raise ValueError(f"Teacher IPC reconstruction requires CUDA, got {device}.")
    if student_seq_start < 0 or student_seq_len <= 0:
        raise ValueError(
            "Invalid student window: "
            f"start={student_seq_start}, length={student_seq_len}."
        )
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    sample_mask = to_local_if_dtensor(data["sample_mask"]).to(device)
    if sample_mask.ndim != 1:
        raise ValueError(
            f"sample_mask must be one-dimensional, got {sample_mask.shape}."
        )

    teacher_full_logits_by_idx: Dict[int, torch.Tensor] = {}
    aligns_by_idx: Dict[int, LocalizedAlignment] = {}
    for i, projection_path in enumerate(projection_matrix_paths):
        ipc_key = f"teacher_{i}_full_logits_ipc"
        if ipc_key not in data:
            raise KeyError(f"Missing x-token teacher IPC field {ipc_key!r}.")
        entries = data[ipc_key]
        teacher_seq_start, teacher_seq_len, teacher_full_seq_len = (
            _teacher_ipc_window_coordinates(
                entries,
                student_cp_rank=student_cp_rank,
                student_cp_size=student_cp_size,
            )
        )
        teacher_logits = rebuild_teacher_full_logits_from_ipc(
            entries,
            cp_group=context_parallel_group,
            device=device_index,
        )
        if teacher_logits.shape[:2] != (sample_mask.shape[0], teacher_seq_len):
            raise ValueError(
                f"Teacher {i} logits have shape {teacher_logits.shape}; expected "
                f"[{sample_mask.shape[0]}, {teacher_seq_len}, V]."
            )
        teacher_full_logits_by_idx[i] = teacher_logits
        if projection_path is None:
            aligns_by_idx[i] = LocalizedAlignment(sample_mask=sample_mask)
            continue

        required_keys = (
            f"teacher_{i}_input_ids",
            f"teacher_{i}_token_mask",
            f"alignment_{i}_student_chunk_id",
            f"alignment_{i}_teacher_chunk_id",
            f"alignment_{i}_pair_valid",
            f"alignment_{i}_pair_is_correct",
        )
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise KeyError(f"Teacher {i} is missing x-token fields: {missing}.")

        teacher_input_ids = to_local_if_dtensor(data[required_keys[0]]).to(device)
        teacher_token_mask_global = to_local_if_dtensor(data[required_keys[1]]).to(
            device
        )
        expected_teacher_shape = (sample_mask.shape[0], teacher_full_seq_len)
        if (
            tuple(teacher_input_ids.shape) != expected_teacher_shape
            or tuple(teacher_token_mask_global.shape) != expected_teacher_shape
        ):
            raise ValueError(
                f"Teacher {i} canonical IDs/mask must have shape "
                f"{expected_teacher_shape}, got {teacher_input_ids.shape} and "
                f"{teacher_token_mask_global.shape}."
            )
        if teacher_input_ids.dtype != torch.long:
            raise TypeError(
                f"Teacher {i} input IDs must have dtype torch.long, got "
                f"{teacher_input_ids.dtype}."
            )
        teacher_next_token_ids_global = teacher_input_ids.roll(-1, dims=1)
        teacher_next_token_mask_global = teacher_token_mask_global.roll(
            -1, dims=1
        ).clone()
        teacher_next_token_mask_global[:, -1] = 0
        teacher_seq_end = teacher_seq_start + teacher_seq_len

        student_chunk_id_global = to_local_if_dtensor(data[required_keys[2]]).to(device)
        if student_chunk_id_global.ndim != 2 or (
            student_seq_start + student_seq_len > student_chunk_id_global.shape[1]
        ):
            raise ValueError(
                f"Teacher {i} student chunk IDs cannot cover student window "
                f"[{student_seq_start}, {student_seq_start + student_seq_len}); "
                f"shape={student_chunk_id_global.shape}."
            )
        student_chunk_id_global = student_chunk_id_global.roll(-1, dims=1).clone()
        student_chunk_id_global[:, -1] = -1

        teacher_chunk_id_global = to_local_if_dtensor(data[required_keys[3]]).to(device)
        if tuple(teacher_chunk_id_global.shape) != expected_teacher_shape:
            raise ValueError(
                f"Teacher {i} chunk IDs must have shape {expected_teacher_shape}, "
                f"got {teacher_chunk_id_global.shape}."
            )
        teacher_chunk_id_global = teacher_chunk_id_global.roll(-1, dims=1).clone()
        teacher_chunk_id_global[:, -1] = -1

        pair_valid = to_local_if_dtensor(data[required_keys[4]]).to(device)
        pair_is_correct = to_local_if_dtensor(data[required_keys[5]]).to(device)
        if (
            pair_valid.shape != pair_is_correct.shape
            or pair_valid.shape[0] != sample_mask.shape[0]
        ):
            raise ValueError(
                f"Teacher {i} pair masks must have matching [B, C] shapes, got "
                f"{pair_valid.shape} and {pair_is_correct.shape}."
            )

        aligns_by_idx[i] = LocalizedAlignment(
            sample_mask=sample_mask,
            student_chunk_id=student_chunk_id_global[
                :, student_seq_start : student_seq_start + student_seq_len
            ].contiguous(),
            teacher_chunk_id=teacher_chunk_id_global[
                :, teacher_seq_start:teacher_seq_end
            ].contiguous(),
            pair_valid=pair_valid,
            pair_is_correct=pair_is_correct,
            teacher_token_mask=teacher_token_mask_global[
                :, teacher_seq_start:teacher_seq_end
            ].contiguous(),
            teacher_next_token_ids=teacher_next_token_ids_global[
                :, teacher_seq_start:teacher_seq_end
            ].contiguous(),
            teacher_next_token_mask=teacher_next_token_mask_global[
                :, teacher_seq_start:teacher_seq_end
            ].contiguous(),
        )
    return teacher_full_logits_by_idx, aligns_by_idx


def prepare_xtoken_window_loss_inputs(
    student_logits_global_sequence: torch.Tensor,
    data: Mapping[str, Any],
    *,
    vocab: ResolvedVocabParallelLogits,
    student_tokenizer_vocab_size: int,
    teacher_tokenizer_vocab_sizes: list[int],
    projection_matrix_paths: list[Optional[str]],
    context_parallel_group: Optional[torch.distributed.ProcessGroup],
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    logprob_chunk_size: Optional[int],
) -> dict[str, Any]:
    """Build the single supported precomputed x-token loss contract.

    ``student_logits_global_sequence`` is already restored from Automodel's
    model-owned token layout. This function owns the x-token contiguous-window
    semantics, global next-token shift, aligned student logprobs, teacher IPC
    reconstruction, and alignment localization.
    """
    if student_logits_global_sequence.ndim != 3:
        raise ValueError(
            "Restored student logits must have shape [B, S, V], got "
            f"{student_logits_global_sequence.shape}."
        )
    if len(teacher_tokenizer_vocab_sizes) != len(projection_matrix_paths):
        raise ValueError(
            "Teacher tokenizer vocab sizes and projection paths must have equal "
            f"lengths, got {len(teacher_tokenizer_vocab_sizes)} and "
            f"{len(projection_matrix_paths)}."
        )
    if vocab.global_vocab_size < student_tokenizer_vocab_size:
        raise ValueError(
            "The student model output width cannot be smaller than the tokenizer "
            f"vocabulary: model={vocab.global_vocab_size}, "
            f"tokenizer={student_tokenizer_vocab_size}."
        )
    local_vocab_width = int(student_logits_global_sequence.shape[-1])
    if local_vocab_width != vocab.vocab_end_index - vocab.vocab_start_index:
        raise ValueError(
            "Restored student logits do not match the resolved vocabulary interval: "
            f"width={local_vocab_width}, interval=[{vocab.vocab_start_index}, "
            f"{vocab.vocab_end_index})."
        )
    if vocab.is_vocab_sharded:
        if vocab.global_vocab_size % vocab.tp_size != 0:
            raise ValueError(
                "x-token requires equal TP vocabulary shards; global vocabulary "
                f"{vocab.global_vocab_size} is not divisible by TP size "
                f"{vocab.tp_size}."
            )
        expected_width = vocab.global_vocab_size // vocab.tp_size
        expected_start = vocab.tp_rank * expected_width
        if (
            local_vocab_width != expected_width
            or vocab.vocab_start_index != expected_start
        ):
            raise ValueError(
                "x-token requires equal, rank-ordered TP vocabulary shards; "
                f"rank {vocab.tp_rank} owns [{vocab.vocab_start_index}, "
                f"{vocab.vocab_end_index}), expected [{expected_start}, "
                f"{expected_start + expected_width})."
            )

    student_cp_rank = (
        torch.distributed.get_rank(context_parallel_group)
        if context_parallel_group is not None
        else 0
    )
    student_cp_size = (
        torch.distributed.get_world_size(context_parallel_group)
        if context_parallel_group is not None
        else 1
    )
    full_sequence_length = int(student_logits_global_sequence.shape[1])
    if full_sequence_length % student_cp_size != 0:
        raise ValueError(
            "x-token requires equal contiguous student CP windows, got sequence "
            f"length {full_sequence_length} and cp_size {student_cp_size}."
        )
    local_sequence_length = full_sequence_length // student_cp_size
    sequence_start = student_cp_rank * local_sequence_length
    sequence_end = sequence_start + local_sequence_length
    student_logits_contig = student_logits_global_sequence[
        :, sequence_start:sequence_end, :
    ].contiguous()

    input_ids = to_local_if_dtensor(data["input_ids"]).to(student_logits_contig.device)
    token_mask = to_local_if_dtensor(data["token_mask"]).to(
        student_logits_contig.device
    )
    if input_ids.shape != token_mask.shape or input_ids.shape != (
        student_logits_contig.shape[0],
        full_sequence_length,
    ):
        raise ValueError(
            "x-token input_ids/token_mask must match the restored student sequence, "
            f"got ids={input_ids.shape}, mask={token_mask.shape}, "
            f"logits={student_logits_global_sequence.shape}."
        )
    next_token_ids = input_ids.roll(-1, dims=1)
    next_token_mask = token_mask.roll(-1, dims=1).clone()
    next_token_ids[:, -1] = 0
    next_token_mask[:, -1] = 0
    student_token_mask_contig = token_mask[:, sequence_start:sequence_end].contiguous()
    student_next_token_ids = next_token_ids[:, sequence_start:sequence_end].contiguous()
    student_next_token_mask = next_token_mask[
        :, sequence_start:sequence_end
    ].contiguous()
    student_next_token_logprobs = get_target_logprobs_from_vocab_parallel_logits(
        student_logits_contig,
        student_next_token_ids,
        tp_group=vocab.vocab_parallel_group,
        vocab_start_index=vocab.vocab_start_index,
        vocab_end_index=vocab.vocab_end_index,
        global_vocab_size=vocab.global_vocab_size,
        chunk_size=logprob_chunk_size,
        sampling_params=None,
        inference_only=False,
    )

    teacher_full_logits_by_idx, aligns_by_idx = (
        _prepare_xtoken_teacher_window_loss_inputs(
            data,
            student_seq_start=sequence_start,
            student_seq_len=local_sequence_length,
            student_cp_rank=student_cp_rank,
            student_cp_size=student_cp_size,
            projection_matrix_paths=projection_matrix_paths,
            context_parallel_group=context_parallel_group,
            device=student_logits_contig.device,
        )
    )
    student_prefix = tuple(student_logits_contig.shape[:2])
    for i, projection_path in enumerate(projection_matrix_paths):
        teacher_logits = teacher_full_logits_by_idx[i]
        align = aligns_by_idx[i]
        if teacher_logits.shape[-1] < teacher_tokenizer_vocab_sizes[i]:
            raise ValueError(
                f"Teacher {i} model output width {teacher_logits.shape[-1]} is "
                "smaller than its tokenizer vocabulary "
                f"{teacher_tokenizer_vocab_sizes[i]}."
            )
        if projection_path is None:
            if tuple(teacher_logits.shape[:2]) != student_prefix:
                raise ValueError(
                    f"Same-vocabulary teacher {i} must align one-to-one with "
                    f"student positions {student_prefix}, got "
                    f"{tuple(teacher_logits.shape[:2])}."
                )
            continue
        assert align.student_chunk_id is not None
        assert align.teacher_chunk_id is not None
        if tuple(align.student_chunk_id.shape) != student_prefix:
            raise ValueError(
                f"Teacher {i} student chunk IDs must match {student_prefix}, got "
                f"{align.student_chunk_id.shape}."
            )
        if tuple(align.teacher_chunk_id.shape) != tuple(teacher_logits.shape[:2]):
            raise ValueError(
                f"Teacher {i} chunk IDs must match teacher positions "
                f"{tuple(teacher_logits.shape[:2])}, got "
                f"{align.teacher_chunk_id.shape}."
            )

    return {
        "student_logits_contig": student_logits_contig,
        "student_output_vocab_size": vocab.global_vocab_size,
        "student_token_mask_contig": student_token_mask_contig,
        "student_next_token_logprobs": student_next_token_logprobs,
        "student_next_token_ids": student_next_token_ids,
        "student_next_token_mask": student_next_token_mask,
        "teacher_full_logits_by_idx": teacher_full_logits_by_idx,
        "aligns_by_idx": aligns_by_idx,
        "tp_group": vocab.vocab_parallel_group,
        "cp_group": context_parallel_group,
        "dp_group": data_parallel_group,
    }
