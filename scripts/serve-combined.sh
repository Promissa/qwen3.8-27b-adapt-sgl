#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL_PATH="${MODEL_PATH:-$PWD/weights/target-embedding-nvfp4}"
export SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1
export SGLANG_GDN_PREFILL_GRAPH_MAX_SEQS=1
exec bash scripts/serve-mtp-graph.sh --max-total-tokens 48128 "$@"
