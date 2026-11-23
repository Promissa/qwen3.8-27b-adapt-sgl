#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime-env.sh
tag="${1:?Usage: bench/run_serving_bench.sh TAG}"
mkdir -p "results/bench/$tag"
.venv/bin/python scripts/environment.py --output "results/bench/$tag/environment.json"
for isl in 1024 8192; do
  for rep in 1 2 3; do
    dest="results/bench/$tag/isl${isl}-rep${rep}.jsonl"
    [[ ! -e "$dest" ]] || { echo "Evidence already exists: $dest" >&2; exit 1; }
    .venv/bin/python -m sglang.benchmark.serving \
      --backend sglang-oai --host 127.0.0.1 --port "${PORT:-30000}" \
      --model "${MODEL_PATH:-$PWD/weights/target}" --served-model-name qwen3.8-27b \
      --tokenizer "${MODEL_PATH:-$PWD/weights/target}" --dataset-name random \
      --random-input-len "$isl" --random-output-len 1024 --random-range-ratio 1 \
      --num-prompts "${NUM_PROMPTS:-10}" --max-concurrency 1 --request-rate inf \
      --flush-cache --output-details --output-file "$PWD/$dest" \
      2>&1 | tee "${dest%.jsonl}.log"
  done
done
