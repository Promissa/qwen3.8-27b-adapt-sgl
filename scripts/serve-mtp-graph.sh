#!/usr/bin/env bash
# Measured C1/D4 Qwen MTP configuration, original target and a 16K token pool.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime-env.sh
export MODEL_PATH="${MODEL_PATH:-$PWD/weights/target}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$PWD/weights/reference-mtp-nvfp4}"
export SGLANG_ENABLE_LOGPROB_CHUNK=1 SGLANG_LOGPROB_CHUNK_SIZE=128
export SGLANG_OPT_QWEN_MTP_EARLY_SHARE=1 SGLANG_OPT_GDN_VERIFY_MATCH_PACKED=1
export SGLANG_OPT_GDN_VERIFY_CONV_ORDER=1 SGLANG_OPT_MHA_VERIFY_GRAPH=1
exec .venv/bin/python scripts/serve.py eagle-native \
  --enable-deterministic-inference --attention-backend triton \
  --linear-attn-prefill-backend triton --mem-fraction-static 0.92 \
  --chunked-prefill-size 1024 --max-mamba-cache-size 4 --max-total-tokens 16384 \
  --enable-metrics "$@"
