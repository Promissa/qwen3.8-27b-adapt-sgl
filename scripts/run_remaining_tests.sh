#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
source scripts/runtime-env.sh
run_dir="${1:?Expected a run directory with frozen benchmark manifests}"
report="$run_dir/remaining-phases.tsv"
if [[ -e "$report" ]]; then
  printf 'Preserve existing results in %s\n' "$run_dir" >&2
  exit 1
fi
printf 'phase\texit_code\tfinished_at\n' >"$report"

run_phase() {
  local phase=$1 rc=0
  shift
  printf '%s started %s\n' "$phase" "$(date -Iseconds)"
  "$@" || rc=$?
  printf '%s\t%s\t%s\n' "$phase" "$rc" "$(date -Iseconds)" >>"$report"
}

if [[ -s "$run_dir/spec/strict-ngram-decode.json" ]]; then
  run_phase spec_divergence .venv/bin/python bench/analyze_token_divergence.py \
    --reference "$run_dir/spec/strict-control.json" \
    --candidate "$run_dir/spec/strict-ngram-decode.json" \
    --output "$run_dir/spec/divergence.json"
fi
run_phase application .venv/bin/python scripts/run_application_tests.py --phase application \
  --manifests "$run_dir" --output "$run_dir/application"
run_phase matched_serving .venv/bin/python scripts/continue_targets.py --phase serving \
  --output "$run_dir/matched-serving"
run_phase recheck .venv/bin/python scripts/run_application_tests.py --phase recheck \
  --manifests "$run_dir" --output "$run_dir/recheck"
