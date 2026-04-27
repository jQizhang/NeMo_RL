# Build vLLM From Source

This guide walks through building vLLM **fully from source** inside the NeMo-RL
uv environment. Use this when no matching precompiled wheel is available — for
example, when tracking a specific unmerged vLLM pull request, a fork, or a
commit that adds new CUDA kernels.

> For the simpler path that reuses a precompiled wheel (fast, no CUDA compile),
> see [Experiment with Custom vLLM](./use-custom-vllm.md). This guide is for
> the cases where that script is not an option.

## When to use this guide

- Tracking a vLLM PR that touches `csrc/` / CMake / new kernels
- Evaluating a fork whose commit hash does not appear on `wheels.vllm.ai`
- The PR requires a specific commit of a transitive kernel library
  (e.g. DeepGEMM) that the pinned NeMo-RL version does not ship

Concrete worked examples:

- Tracking vLLM `main` directly (used for the current `dsv4-support` branch — DeepSeek-V4
  support landed on `main` after [PR #40760](https://github.com/vllm-project/vllm/pull/40760)
  / [PR #40860](https://github.com/vllm-project/vllm/pull/40860) merged).
- Tracking a still-open PR (e.g. an unmerged kernel change) — substitute the PR
  number in step 1.

## Prerequisites

- A running GPU node with NVCC matching the project's CUDA version (CUDA 12.9
  for torch `2.10.0+cu129`). Check with `nvcc --version`.
- The NeMo-RL container (or an equivalent env with Python 3.13, `uv`, and the
  project venv already bootstrapped — typically `UV_PROJECT_ENVIRONMENT` is set
  by the container).
- ≥ 20 GB free disk for the vLLM source checkout + build artifacts.

Sanity check before starting:

```sh
cd "$NRL_ROOT"
echo "uv:        $(uv --version)"
echo "python:    $(uv run python --version)"
echo "torch:     $(uv run python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
nvcc --version | tail -1
```

## Step 1 — Fetch the vLLM source

Create a workspace directory inside `3rdparty/` and check out the branch you
want to build against. The default below tracks `main`; adapt the
`git checkout` line if you need a specific PR, branch, or SHA.

```sh
cd "$NRL_ROOT"
mkdir -p 3rdparty/vllm-workspace
git clone --filter=blob:none https://github.com/vllm-project/vllm.git \
    3rdparty/vllm-workspace/vllm

cd 3rdparty/vllm-workspace/vllm

# Default: track upstream main
git checkout main
git pull origin main

# Or a specific PR:
# git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
# git checkout pr-<PR_NUMBER>

# Or a specific branch / SHA:
# git checkout <branch-or-sha>

git rev-parse HEAD   # record this; it will appear in vllm.__version__
```

If you already have the workspace checked out from a previous PR build,
reuse it — just fetch and reset to the new ref, and clean any stale
compiled extensions so `uv sync` rebuilds cleanly:

```sh
cd "$NRL_ROOT/3rdparty/vllm-workspace/vllm"
git fetch origin
git checkout main && git pull --ff-only origin main
# Remove stale .so artifacts from the previous build so the new one is fresh
rm -f vllm/*.so
rm -rf build/
```

Add the workspace directory to `.gitignore` so the 4 GB+ checkout is not
committed:

```
3rdparty/vllm-workspace/
```

## Step 2 — Wire vLLM into `pyproject.toml`

Point the uv resolver at the local path and tell it how to build without
isolation (so the runtime torch is used at build time, not a freshly resolved
build-env torch).

Add to `[tool.uv.sources]`:

```toml
vllm = { path = "3rdparty/vllm-workspace/vllm", editable = true }
```

Drop the version pin from the `vllm` extra and leave just the name
(uv will resolve against the path source):

```toml
[project.optional-dependencies]
vllm = [
  # ...other deps...
  "vllm",
  # ...
]
```

Add `vllm` to the no-build-isolation list:

```toml
[tool.uv]
no-build-isolation-package = [
  # ...
  "vllm",
]
```

Add vLLM's build-system requirements to `[tool.uv.extra-build-dependencies]`,
**except** torch — torch comes from the runtime venv:

```toml
[tool.uv.extra-build-dependencies]
vllm = [
  "cmake>=3.26.1",
  "ninja",
  "packaging>=24.2",
  "setuptools>=77.0.3,<81.0.0",
  # setuptools-scm 10.x split out vcs_versioning which breaks the
  # prepare_metadata_for_build_editable hook. Cap below 9.
  "setuptools-scm>=8.0,<9",
  "wheel",
  "jinja2",
]
```

Also add `cmake` and `setuptools-scm<9` to the default `build`
dependency-group so `uv sync` installs them into the project venv (needed
because, with `no-build-isolation`, vllm's setup.py imports them from the
venv directly):

```toml
[dependency-groups]
build = [
  # ...existing build deps...
  "cmake>=3.26.1",
  "setuptools-scm>=8.0,<9",
]
```

## Step 3 — Reconcile vLLM's runtime dependencies

vLLM's `requirements/cuda.txt` will likely pin newer versions of several
packages than NeMo-RL. Use `[tool.uv.override-dependencies]` to keep NeMo-RL's
pins authoritative where the torch ABI matters (torch, torchvision,
torchaudio), and bump lower-bound overrides where vLLM genuinely needs a
newer release (e.g. `nvidia-cutlass-dsl>=4.4.2`).

Rule of thumb:

| vLLM wants | NeMo-RL has | Action |
|---|---|---|
| Newer torch (e.g. `2.11`) | `torch==2.10.0` | Keep NeMo-RL pin via `override-dependencies` |
| Newer torchvision / torchaudio | NeMo-RL pin | Same — pin to NeMo-RL's torch stack |
| Newer `nvidia-cutlass-dsl` / `flashinfer-python` | Older | Bump the NeMo-RL override |
| `transformers >= 4.56, != 5.0.*..!= 5.5.0` | Pinned at `5.3.0` via override | Keep `5.3.0` — `override-dependencies` replaces vLLM's exclusion (vLLM code has been v5-compatible since 0.17) |
| New transitive runtime deps (`quack-kernels`, `apache-tvm-ffi`, `tilelang`, `fastsafetensors`, …) | Not present | No action — uv resolves them automatically from vLLM's `requirements/cuda.txt` |

Inspect `3rdparty/vllm-workspace/vllm/requirements/common.txt` and
`requirements/cuda.txt`; diff against `[project.optional-dependencies].vllm`
and `[tool.uv.override-dependencies]` in NeMo-RL's `pyproject.toml` and
decide each version conflict explicitly.

## Step 4 — Align transitive kernel libraries (DeepGEMM, etc.)

vLLM vendors kernel libraries via CMake's `FetchContent_Declare` — inspect the
vLLM source's `CMakeLists.txt`, included `cmake/*.cmake`, and
`cmake/external_projects/*.cmake` for pinned commits. If NeMo-RL's
`pyproject.toml` ships a **different** commit of the same library, symbols
expected by vLLM may be missing at runtime, with errors like:

```
RuntimeError: DeepGEMM backend is not available or outdated.
Please install or update the `deep_gemm` to a newer version to enable FP8 kernels.
```

Find the vLLM-expected commit:

```sh
grep -A 3 "FetchContent_Declare" 3rdparty/vllm-workspace/vllm/CMakeLists.txt \
    3rdparty/vllm-workspace/vllm/cmake/*.cmake \
    3rdparty/vllm-workspace/vllm/cmake/external_projects/*.cmake \
    2>/dev/null | grep -B 1 GIT_TAG
```

For DeepGEMM specifically, the pin lives in
`cmake/external_projects/deepgemm.cmake` and is mirrored in
`tools/install_deepgemm.sh` — both must agree.

Then update both the dependency and its metadata block in NeMo-RL's
`pyproject.toml`. Example for DeepGEMM:

```toml
# In [project.optional-dependencies].vllm
"deep_gemm @ git+https://github.com/deepseek-ai/DeepGEMM.git@<vllm-cmake-pin>"

# In the matching [[tool.uv.dependency-metadata]] block
version = "v2.0.0+<first-7-chars-of-pin>"
```

## Step 5 — Lock

The vLLM path source uses `no-build-isolation`, so its `setup.py` runs against
the project venv directly. `prepare_metadata_for_build_editable` imports
`setuptools_scm` and `cmake` at metadata time — pre-install them so the first
`uv lock` doesn't fail:

```sh
uv pip install --no-deps "setuptools-scm<9" cmake
```

(The `<9` cap is because `setuptools-scm` 10.x split `vcs_versioning` into a
separate distribution that the metadata hook does not pull in.)

Then:

```sh
uv lock
```

This step only reads metadata — it does not build CUDA kernels. Expected
warnings about extras (`transformer-engine ... extra named 'core-cu12'`,
`nvidia-modelopt ... extra named 'torch'`) are benign.

## Step 6 — Sync (this is where CUDA kernels compile)

```sh
uv sync --extra vllm
```

What happens:

1. uv installs/upgrades any packages that moved in the lock.
2. With `no-build-isolation`, uv invokes vLLM's `setup.py` inside the project
   venv. vLLM's CMake driver (`setup.py` → `cmake_build_ext`) compiles all
   `csrc/` kernels. This produces `vllm/_C.abi3.so` (~440 MB) and
   `vllm/_C_stable_libtorch.abi3.so` (~220 MB).
3. Any transitive C++/CUDA libraries marked `no-build-isolation-package`
   (DeepGEMM, flash-attn, transformer-engine, …) are also rebuilt if their
   pinned source changed in step 4.

**Expected timings on 8 × H100, warm Lustre cache:**

| Step | Time |
|---|---|
| vLLM full compile | ~15 min |
| DeepGEMM rebuild (after version bump) | ~45 s |
| Full `uv sync --extra vllm` (first time, cold) | ~20 min |

## Step 7 — Verify

```sh
uv run --extra vllm python -c \
  'import vllm; print(vllm.__version__); print(vllm.__file__)'
```

Expected output:

- `vllm.__version__` contains a git suffix derived from setuptools-scm, e.g.
  `0.19.1rc1.dev249+gfe61cd4da` — the suffix after `+g` matches the SHA you
  checked out in step 1.
- `vllm.__file__` resolves inside
  `3rdparty/vllm-workspace/vllm/vllm/__init__.py`, confirming the editable
  install is live.

Confirm the expected transitive kernels loaded:

```sh
uv run --extra vllm python -c \
  'import deep_gemm, flashinfer; print("deep_gemm:", deep_gemm.__file__); print("flashinfer:", flashinfer.__version__)'
```

If you bumped DeepGEMM for a feature that requires a specific symbol (e.g.
`tf32_hc_prenorm_gemm` for the original DeepSeek-V4 PR), spot-check that it
is exported:

```sh
uv run --extra vllm python -c \
  'import deep_gemm; print(hasattr(deep_gemm, "tf32_hc_prenorm_gemm"))'
```

Run a smoke inference (OpenAI-compatible server or offline `LLM.generate()`
— whichever best matches your model). Note: on first use, DeepGEMM JIT-compiles
per-shape kernels (first request on a fresh cache can take ~5 min for
JIT warmup; subsequent requests hit the JIT cache).

## Troubleshooting

### `ModuleNotFoundError: No module named 'setuptools_scm'` during `uv lock`

uv cannot read vLLM's dynamic metadata because `setup.py` imports
`setuptools_scm` and the venv does not have it. `extra-build-dependencies`
does not auto-install for `no-build-isolation` packages; install manually:

```sh
uv pip install --no-deps "setuptools-scm<9"
```

### `ModuleNotFoundError: No module named 'vcs_versioning'`

`setuptools-scm==10.x` split the `vcs_versioning` namespace into a separate
distribution which the metadata hook does not pull in. Pin below 9 (see the
`extra-build-dependencies` and `build` group examples in step 2).

### `RuntimeError: DeepGEMM backend is not available or outdated`

The installed `deep_gemm` is older than the commit vLLM was built against.
Follow step 4 to align the commit, then:

```sh
uv lock && uv sync --extra vllm
```

### `RuntimeError: Engine core initialization failed` with `freeze_support()` hint

Your entrypoint script is missing the `if __name__ == "__main__":` guard.
vLLM forces `spawn` when CUDA is already initialized; the child processes
re-import the entrypoint module and must not re-execute `LLM(...)` at import
time. Wrap top-level code in `main()` and guard it.

### Shape mismatches during `_load_w13` / MoE weight load

The checkpoint's tensor layout does not match what the vLLM model loader
expects (for example, standard FP8 `(moe_intermediate_size, hidden_size)` vs.
an INT8-packed variant). This is usually a model–vLLM compatibility gap on
the vLLM side, not a build issue. Check the PR discussion or file an upstream
issue.

## Reverting to the stock vLLM wheel

```sh
# Restore pyproject.toml from main:
git checkout main -- pyproject.toml uv.lock

# Re-sync (uv will reinstall the pinned PyPI vllm):
uv sync --extra vllm

# Optionally remove the source checkout:
rm -rf 3rdparty/vllm-workspace/
```
