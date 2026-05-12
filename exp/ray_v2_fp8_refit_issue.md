# vLLM Ray V2 Executor 导致 FP8 Refit 失败的问题分析

## 结论

在 GB200 container 中遇到的 `NoneType.use_weight_pow2_scale` 报错，直接原因是当前 vLLM 默认启用了 Ray V2 executor：

```text
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=True
```

实际运行路径变成：

```text
RayExecutorV2 -> MultiprocExecutor.collective_rpc
```

而 NeMo-RL 当前 FP8 refit patch 只覆盖旧路径：

```text
vllm.v1.executor.ray_executor.RayDistributedExecutor.collective_rpc
```

因此 vLLM 的 TP worker 没有执行 `apply_fp8_patches(fp8_config)`，导致 `nemo_rl.models.generation.vllm.quantization.fp8.global_fp8_config` 在 TP worker 进程中仍然是 `None`。当 refit 逻辑把 BF16 policy weights 重新 cast 成 FP8 weights 时，访问：

```python
global_fp8_config.use_weight_pow2_scale
```

就会触发：

```text
AttributeError: 'NoneType' object has no attribute 'use_weight_pow2_scale'
```

## 报错现象

典型日志如下：

```text
VllmGenerationWorker pid=621940) Error: Worker failed to update weights. Result: False

RayWorkerProc pid=622848) (Worker_TP0 pid=622848)
Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq:
'NoneType' object has no attribute 'use_weight_pow2_scale'

File "/apps/nemorl-ds4/nemo_rl/models/generation/vllm/vllm_backend.py", line 255, in update_weights_via_ipc_zmq
  fp8.load_weights(policy_weights, self.model_runner)

File "/apps/nemorl-ds4/nemo_rl/models/generation/vllm/quantization/fp8.py", line 666, in cast_tensor_to_fp8_blockwise
  if global_fp8_config.use_weight_pow2_scale:

AttributeError: 'NoneType' object has no attribute 'use_weight_pow2_scale'
```

这说明：

1. policy weights 已经通过 ZMQ/CUDA IPC 传到了 vLLM worker。
2. vLLM 进入了 FP8 refit 分支。
3. 失败发生在 FP8 weight cast 阶段，不是模型初始化阶段，也不是 CUDA IPC handle 阶段。

## 运行环境验证

在 GB200 container 中验证到：

```text
GPU: NVIDIA GB200, capability (10, 0)
vLLM path: /apps/nemorl-ds4/3rdparty/vllm-workspace/vllm/vllm
NeMo-RL path: /apps/nemorl-ds4/nemo_rl
FP8 module: /apps/nemorl-ds4/nemo_rl/models/generation/vllm/quantization/fp8.py
```

vLLM 默认 executor 配置：

```python
import vllm.envs as envs
print(envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND)
print(envs.VLLM_ENABLE_V1_MULTIPROCESSING)
```

输出：

```text
VLLM_USE_RAY_V2_EXECUTOR_BACKEND True
VLLM_ENABLE_V1_MULTIPROCESSING True
```

Ray 日志也确认实际使用了 Ray V2 executor：

```text
ERROR ... [ray_executor_v2.py:464] RayWorkerProc rank=[3] died unexpectedly, shutting down executor.
```

## Patch 覆盖范围验证

当前 NeMo-RL FP8 patch 位于：

```text
nemo_rl/models/generation/vllm/quantization/fp8.py
```

核心逻辑是：

```python
from vllm.v1.executor.ray_executor import RayDistributedExecutor
RayDistributedExecutor.collective_rpc = patched_run_workers
```

在 GB200 container 中验证 patch 前后状态：

```text
before RayDistributedExecutor.collective_rpc
  vllm.v1.executor.ray_executor RayDistributedExecutor.collective_rpc

before MultiprocExecutor.collective_rpc
  vllm.v1.executor.multiproc_executor MultiprocExecutor.collective_rpc

RayExecutorV2 mro
  ['RayExecutorV2', 'MultiprocExecutor', 'Executor', 'ABC']

after RayDistributedExecutor.collective_rpc
  nemo_rl.models.generation.vllm.quantization.fp8 ... patched_run_workers

after MultiprocExecutor.collective_rpc
  vllm.v1.executor.multiproc_executor MultiprocExecutor.collective_rpc

RayExecutorV2.collective_rpc
  vllm.v1.executor.multiproc_executor MultiprocExecutor.collective_rpc
```

含义：

- 旧 `RayDistributedExecutor.collective_rpc` 被 patch 成功。
- 新 `RayExecutorV2` 继承的是 `MultiprocExecutor.collective_rpc`。
- `MultiprocExecutor.collective_rpc` 没有被 NeMo-RL patch。
- 所以 Ray V2 路径下 TP worker 不会自动执行 `apply_fp8_patches(fp8_config)`。

## 为什么 H100 上可以正常工作

这个问题本质上是 vLLM executor 路径差异，不是 GB200 GPU 架构本身导致的。

H100 上之前可以正常工作，最可能原因是 H100 环境满足以下条件之一：

1. 使用了旧版本 vLLM，默认 executor 是 `RayDistributedExecutor`。
2. 环境变量显式或隐式设置了：

```bash
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
```

3. vLLM 代码分支还没有切到 `RayExecutorV2` 默认路径。

如果 H100 也使用同一套新 vLLM 代码，并启用：

```bash
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
```

同时运行 `precision: fp8` 的 vLLM refit 路径，大概率也会遇到同样的 `global_fp8_config is None` 报错。

## 临时 Workaround

在启动训练脚本前强制关闭 Ray V2 executor：

```bash
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
```

例如放在 `run_dsv4_4l_grpo.sh` 中，`uv run examples/run_grpo.py` 之前：

```bash
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0

uv run examples/run_grpo.py \
    --config exp/grpo-dsv4-flash-4layers-1n8g.yaml
```

预期效果：

- vLLM 回到旧 `RayDistributedExecutor`。
- NeMo-RL 当前 `monkey_patch_vllm_ray_executor()` 能命中。
- TP workers 会在第一次 collective RPC 前执行 `apply_fp8_patches(fp8_config)`。
- `global_fp8_config` 不再是 `None`。

## 长期修复方向

应修改 `nemo_rl/models/generation/vllm/quantization/fp8.py`，让 FP8 patch 同时覆盖 Ray V2 executor。

当前只覆盖：

```text
vllm.v1.executor.ray_executor.RayDistributedExecutor.collective_rpc
```

需要新增覆盖：

```text
vllm.v1.executor.ray_executor_v2.RayExecutorV2
vllm.v1.executor.multiproc_executor.MultiprocExecutor.collective_rpc
```

Ray V2 的控制面通过 `MessageQueue` 广播 RPC：

```text
MultiprocExecutor.collective_rpc
  -> rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
  -> WorkerProc.worker_busy_loop()
  -> cloudpickle.loads(method)(self.worker)
```

因此可以考虑在 `RayExecutorV2`/`MultiprocExecutor` 的第一次 `collective_rpc` 前，向所有 workers 广播执行：

```python
apply_fp8_patches(fp8_config)
```

注意事项：

- 需要避免重复 patch，同进程内仍然应使用 `fp8_patches_applied` guard。
- 需要保证 patch 执行在 `update_weights_via_ipc_zmq()` 之前。
- 对旧 `RayDistributedExecutor` 的 patch 保持兼容。
- 对非 Ray 或单 GPU 路径保持现有行为。

## 验证建议

### 验证 workaround

在 GB200 container 中设置：

```bash
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
```

重跑同一脚本。如果 `NoneType.use_weight_pow2_scale` 消失，说明问题确实来自 Ray V2 executor patch 未覆盖。

### 验证 H100/GB200 差异

在 H100 上强制启用：

```bash
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
```

使用同一 vLLM 代码和同一 NeMo-RL 分支运行。如果复现同样报错，说明问题与 GPU 类型无关，而是 executor 路径差异。

### 验证长期修复

在 vLLM TP worker 中打印：

```python
from nemo_rl.models.generation.vllm.quantization import fp8
print("fp8 file", fp8.__file__)
print("global_fp8_config", fp8.global_fp8_config)
print("fp8_patches_applied", fp8.fp8_patches_applied)
```

修复后，在 `update_weights_via_ipc_zmq()` 调用 `fp8.load_weights()` 前应满足：

```text
global_fp8_config is not None
fp8_patches_applied is True
```

## 相关但不同的问题

之前遇到的：

```text
RuntimeError: pidfd_getfd: Bad file descriptor
```

是 CUDA IPC handle rebuild 阶段的问题，和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` / CUDA allocator / GB200 环境相关性更高。

本文件分析的：

```text
AttributeError: 'NoneType' object has no attribute 'use_weight_pow2_scale'
```

发生在 CUDA IPC 已成功之后的 FP8 refit cast 阶段，直接原因是 Ray V2 executor 下 FP8 patch 没有传播到 TP worker。
