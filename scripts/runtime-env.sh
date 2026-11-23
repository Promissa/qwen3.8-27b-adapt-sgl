#!/usr/bin/env bash
# Keep generated caches in the workspace. Safe to source from any directory.
if [[ -n "${BASH_VERSION:-}" ]]; then
  task_source="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  task_source="${(%):-%x}"
else
  echo 'Source runtime-env.sh from bash or zsh.' >&2
  return 1
fi
task_root=$(cd "$(dirname "$task_source")/.." && pwd)
if [[ -d "$task_root/engine/sglang/python/sglang" ]]; then
  export SGLANG_SOURCE="${SGLANG_SOURCE:-$task_root/engine/sglang}"
  export PYTHONPATH="$SGLANG_SOURCE/python${PYTHONPATH:+:$PYTHONPATH}"
fi
export PATH="$task_root/.venv/bin:$PATH"
if [[ -x "$task_root/.venv/bin/python" ]]; then
  task_ninja_bin=$("$task_root/.venv/bin/python" - <<'PY'
try:
    import ninja
    from pathlib import Path
    import importlib.metadata
    root = Path(ninja.BIN_DIR) if ninja.BIN_DIR else None
    if root is None or not (root / 'ninja').is_file():
        executable = importlib.metadata.distribution('ninja').locate_file('../../../bin/ninja').resolve()
        root = executable.parent if executable.is_file() else None
    if root is not None:
        print(root)
except ImportError:
    pass
PY
  )
  if [[ -n "$task_ninja_bin" ]]; then
    # Ninja may live in the shared serving venv, whose Python lacks project
    # evaluation packages. Keep the project interpreter ahead of that directory.
    export PATH="$task_root/.venv/bin:$task_ninja_bin:$PATH"
  fi
fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$task_root}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$task_root/.cache/sglang}"
# Reuse the user's existing Hub credential by path; keep all new cache data local.
task_hf_token="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [[ -z "${HF_TOKEN_PATH:-}" && -f "$task_hf_token" ]]; then
  export HF_TOKEN_PATH="$task_hf_token"
fi
export HF_HOME="${HF_HOME:-$task_root/.cache/huggingface}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$task_root/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$task_root/.cache/triton}"
export MAX_JOBS="${MAX_JOBS:-2}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
if [[ -z "${CUDA_HOME:-}" && -x "$task_root/.venv/bin/python" ]]; then
  task_cuda_root=$("$task_root/.venv/bin/python" "$task_root/scripts/prepare_cuda.py" 2>/dev/null || true)
  if [[ -n "$task_cuda_root" ]]; then
    export CUDA_HOME="$task_cuda_root"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
