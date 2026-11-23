#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime-env.sh
export MODEL_PATH="${MODEL_PATH:-$PWD/weights/target-embedding-nvfp4}"
export SGLANG_ENABLE_LOGPROB_CHUNK="${SGLANG_ENABLE_LOGPROB_CHUNK:-1}"
export SGLANG_LOGPROB_CHUNK_SIZE="${SGLANG_LOGPROB_CHUNK_SIZE:-128}"
exec .venv/bin/python scripts/serve.py balanced \
  --enable-metrics --max-mamba-cache-size 4 --max-total-tokens 49152 "$@"
