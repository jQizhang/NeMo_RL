# GB200 GRPO training hangs around step 3-4 during DTensor policy offload/refit

**Describe the bug**

GRPO training for DeepSeek V4 Flash hangs on GB200 around step 3-4. The job can finish the first few steps, then stops making progress during the next generation/refit cycle and eventually either needs to be cancelled or fails after a NCCL watchdog timeout.

Recent debug log from `2638324-logs/ray-driver.log` shows the hang is likely in the policy worker post-refit CPU offload path:

```text
DTensorPolicyWorkerV2[rank=5]: offload_after_refit/model_to_cpu/move_to_device_start ...
DTensorPolicyWorkerV2[rank=5]: offload_after_refit/model_to_cpu/move_to_device_done after 4106.03s ...

DTensorPolicyWorkerV2[rank=1]: offload_after_refit/model_to_cpu/move_to_device_start ...
DTensorPolicyWorkerV2[rank=1]: offload_after_refit/model_to_cpu/move_to_device_done after 5133.73s ...

[rank3] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=64214, OpType=ALLGATHER, NumelIn=1, NumelOut=32, Timeout(ms)=600000)
```

The same script previously ran on H100, but repeatedly hangs on GB200.

**Steps/Code to reproduce bug**

Run 8-node GRPO training for DeepSeek V4 Flash on GB200 nodes, with 4 GPUs per node:

The run uses:

```text
model: Deepseek/DeepSeek-V4-Flash-Base
policy backend: AutoModel DTensor/FSDP2
generation backend: vLLM, FP8, DeepGEMM enabled
nodes: 8
gpus_per_node: 4
```

<details>
<summary>Relevant YAML config</summary>

```yaml
grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  max_num_steps: 10000
  max_response_length: 2048

policy:
  model_name: Deepseek/DeepSeek-V4-Flash-Base
  train_micro_batch_size: 1
  train_global_batch_size: 512
  logprob_batch_size: 1
  precision: bfloat16
  max_total_sequence_length: 4096
  logprob_chunk_size: 1024
  offload_optimizer_for_logprob: true
  dtensor_cfg:
    expert_parallel_size: 32
    activation_checkpointing: true
    offload_after_refit_grouping: node_local_rank
    moe_parallelizer:
      ignore_router_for_ac: true
    env_vars:
      NEMO_AUTOMODEL_DSV4_EXPERT_LAYOUT: base
      NEMO_AUTOMODEL_MOE_SCATTER_CHUNK_ROWS: "4096"
    automodel_kwargs:
      backend:
        attn: sdpa
        linear: torch
        rms_norm: torch_fp32
        dispatcher: torch
        experts: torch_mm
        enable_hf_state_dict_adapter: true
  generation:
    max_new_tokens: 2048
    vllm_cfg:
      tensor_parallel_size: 8
      pipeline_parallel_size: 1
      expert_parallel_size: 16
      enable_expert_parallel: true
      precision: fp8
      kv_cache_dtype: fp8_ds_mla
      pow2_weight_scaling_factors: true
      gpu_memory_utilization: 0.6
      max_model_len: 4096
      use_deep_gemm: true
    colocated:
      enabled: true

cluster:
  gpus_per_node: 4
  num_nodes: 8
```

</details>

**Expected behavior**

Training should continue normally past step 4. For this repro, running at least 10 steps without hanging would be considered successful.

**Additional context**

This was reproduced multiple times:

```text
2635901: reached step 3, then stopped making progress
2636224: reached step 3, then stopped making progress
2637238: reached step 3, then stopped making progress
2637499: reached step 4, then stopped making progress until cancelled
2638324: debug run showed very slow policy offload/refit and then NCCL timeout
```

The likely hang point is:

```text
DTensorPolicyWorkerV2.offload_after_refit()
  -> self.move_to_cpu(self.model)
  -> model.to("cpu")
```
