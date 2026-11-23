# Reused experiment material

Source directory: `/home/pr/projects/qwen38-27b-rtx3090/bench` (read-only).

- `prompts_real.jsonl`: copied unchanged.
- `seat_ttft.py`: adapted; default port changed from 18020 to30000 and usage
  interpreter changed to `.venv/bin/python`. TTFT uses the first content SSE
  chunk and actual usage counts; context size remains approximate. The initial
  sample is now labeled `first_in_series`, since this client cannot establish
  a fresh server. Even-sized warm samples use the statistical median rather
  than the upper middle value. Mock transport/timestamp verification is retained
  in `results/tests-seat-ttft-protocol.txt`; it is not a live timing result.

- `conc_ladder.py`: adapts the old client's SSE first/last-content timing and
  actual usage counts to the reused real prompts. Runs all eight prompts in every repetition, retains their corpus hash and
  per-run salts, keeps every repetition, and reports
  actual simultaneous decode overlap, captures raw SGLang metrics and leaves
  missing series unknown. It does not equate scheduler iterations with forwards
  or completion counters with live decode tokens. GPU validation is pending.
- `quality_battery.py`: adapts the prior EN/DA/code 40-window/domain perplexity
  protocol, using SGLang's actual `input_token_logprobs`. Code windows come from
  the fixed SGLang source rather than vLLM. The old 200-example greedy nonthinking
  GSM8K subset is also retained, separately from the official 1319-example
  thinking GSM8K run. Prior baseline evidence covers perplexity only.
  GPQA Diamond was attempted with the user's existing credential and is gated.
