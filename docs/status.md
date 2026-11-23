# Execution status

## November 21 finalization (Everyday-PC scope)

- **Scope boundary:** Everyday PC usage with desktop sharing the GPU; absent-desktop and Windows Sysmem Fallback enforcement are formally dropped. Recommended launchers pin safe `--max-total-tokens` ceilings.
- **De-tax 4-arm ablation (`results/detax-20261121/comparison.json`):** At 48,128 pool (`patch 0008` off), deterministic Triton + `0007` (chunk 1024) achieved **729.8 ms** 8K cold TTFT and **157.2 tok/s** 8K decode (100% bit-exact), outperforming chunk 2048 (734.1 ms / 158.4 tok/s), non-deterministic FlashInfer MTP (766.7 ms / 153.1 tok/s, 0/4 exact), and FlashInfer GDN prefill (828.9 ms / 144.2 tok/s, 0/4 exact). `patch 0008` was removed from `scripts/serve-combined.sh`.
- **Finalized `serve-combined.sh` paired evaluation (`results/combined-eval-20261121/`):**
  - API smoke: 5/5 checks passed (`tool_calls`, `reasoning_content`, `thinking_disabled`, `reasoning_effort`, `thinking_sampled`).
  - Serving (3-repeat medians): 1K **150.20 tok/s** (86.55 ms TTFT, 6.898 s e2e, **2.32×** vs Reference); 8K **157.74 tok/s** (729.15 ms TTFT, 7.215 s e2e, **2.32×** vs Reference).
  - Near-limit retrieval (47,608 input, 3 depths): **3/3 passed**, 47,616 cached tokens hit, warm TTFT **112.0–114.4 ms**.
  - Paired BFCL (31 common w/ Reference): **14/31** vs Reference 13/31 (+1) and Balanced 18/31; total wall time **313.8 s** vs Reference 513.8 s / Balanced 524.0 s (**1.64×–1.67×** real-world multi-turn speedup).
  - Paired LongBench v2 (16 total): **5/16** (8 capped) vs Reference 6/16 (8 capped) and Balanced 7/16 (5 capped).
- **IFBench 16K truncation attribution (`results/ifbench-trunc-diag-20261121/summary.json`):** Re-running a `seed=42` sample of the 169 prompts previously truncated at `max_tokens=4096` (18/20 completed in 1,800 s) with `max_tokens=16384` recovered **10/18 (55.56% strict & loose)** from **0/18 (0.0%)** at 4,096 (13/18 natural stops, **76.92%** pass rate; 5/18 still hit 16,384; mean/median 9,685 / 10,727 tokens). Extrapolated full-300 score at 16K budget is **~69.0% loose / ~68.0% strict**, confirming that the 39.33% headline at 4,096 was driven by thinking-budget truncation rather than quantization or MTP degradation.

The [80%-context study](profiling.md#实测上下文极限与-80-输入) measures capacity with adjacent accepted and rejected requests and evaluates the application outputs alongside latency. A 48,128-token MTP pool achieved 1.845× on 45,956 input plus 1,024 output tokens, with both generated sequences identical to their controls. The 57,452-token pool measured 0.552× on the same inputs under higher desktop occupancy.

The [accelerated screening results](profiling.md#11-月-19-日加速筛查实测) include a matched CUDA Graph MTP comparison, cold and warm prefill tests and an ordinary decode ablation. The [raw comparison](../results/fast-20261119/comparison.json) reports the two-round means.

The November 19 continuation uses the frozen BFCL and LongBench v2 subsets
in [PLAN.md](../PLAN.md). Its [run directory](../results/continuation-20261119)
contains the frozen IDs, paired evidence and phase log.
The [execution log](../results/continuation-20261119/queue.log) records phase
starts, and the [phase table](../results/continuation-20261119/remaining-phases.tsv)
records exit codes as phases finish.
[Live summary](../results/continuation-20261119/progress.json) reports current
profile results and soak durations.

- The project test suite passed all 108 checks in the serving environment.
- The fixed-pool A/B/B/A run passed all four inference checks. Device-wide
  memory at readiness fell by 1.45 GiB on average. Subtracting the pre-launch
  reading gives 1.58 GiB. Desktop memory drift leaves the 1.5 GiB target
  inconclusive for the 4K pool. The separate 16K-pool A/B/B/A replay shows
  1.64 GiB device-wide savings and 1.64–1.77 GiB after idle correction.
  It measured candidate/control decode ratios of 0.990× and 0.985× at
  1K and 8K input, with six requests per model and length.
  [4K memory](../results/continuation-20261119/memory/comparison.json)
  · [16K memory and speed](../results/continuation-20261119/matched-serving/comparison.json)
- Experimental eager NGRAM with verify routed through decode matched all
  eight 512-token outputs. Aggregate end-to-end speedup on those requests was
  1.14×. The 1.5× target remains unmet in this test.
  [Strict timing and acceptance](../results/continuation-20261119/spec/latency-comparison.json)
- The Windows fallback read still returned NVAPI −160 with the NVIDIA display
  active. The matched physical-context gain still lacks the required host
  condition. [Readback](../results/continuation-20261119/windows-fallback-read.json)
- The application reference passed 13 of 31 scored BFCL conversations and
  6 of 16 LongBench questions. One BFCL task reached 101,791 tokens during the
  conversation and was rejected at the 49,146-token server limit. The official
  scorer recovered the other 23 holdout results from saved responses. Eight
  LongBench responses reached the 128-token output budget.
  [BFCL holdout scoring](../results/continuation-20261119/application/reference/bfcl-holdout-rescore/summary.json)
- Balanced passed 18/31 common BFCL tasks versus the reference's 13/31 and
  7/16 LongBench questions versus 6/16. Two tasks in each benchmark regressed
  and are selected for the targeted repeat. The three-repeat native decode
  medians were 63.08/62.56 tok/s at 1K/8K input, with cold TTFT of
  86.7/687.9 ms. Retrieval passed at 48,640 tokens at all three depths.
  Warm prefix-hit TTFT was 53.8–117.8 ms.
  [Paired scores](../results/continuation-20261119/application/paired-quality.json)
  · [Serving](../results/continuation-20261119/application/balanced/serving.json)
  · [Retrieval](../results/continuation-20261119/application/balanced/retrieval.json)
- Max-context completed the same subsets at 16/31 common BFCL tasks and 6/16
  LongBench questions. Three BFCL and two LongBench tasks went from passing
  to failing. Both candidate profiles hit the LongBench output budget on five
  questions, compared with eight on the reference. The three-repeat native
  decode medians were 64.97/63.42 tok/s at 1K/8K input, with cold TTFT of
  82.6/631.7 ms. Retrieval passed at 81,400 tokens at all three depths.
  Repeat-request TTFT was 14.70–14.73 s with radix disabled.
  [Serving](../results/continuation-20261119/application/max-context/serving.json)
  · [Retrieval](../results/continuation-20261119/application/max-context/retrieval.json)
- The targeted Balanced rerun passed both disputed BFCL tasks. It reproduced
  one of the two disputed LongBench failures while the reference passed both.
  The user subsequently cancelled the stability runs. Balanced stopped at
  1,169 seconds, and the Max-context soak was removed before launch.
  [Cancellation record](../results/continuation-20261119/soak/cancellation.json)

## November 12 reduced delivery

The user-authorized three-hour reduced delivery is complete. All10 planned
serial trials finish, with320 new thinking examples, and the final audit has no
missing evidence. The final audit verified that port30000 was free. See the
[delivery report](three-hour-delivery.md) for actual completion time and every
result; the original positive targets remain distinct from this revised scope.

## Selected configurations

The named scripts now default to converted embedding weights, FP8 KV and BF16
state, with experimental speculative switches off. Their expanded flags and
logprob defaults match the passed probes exactly.

- Balanced:49,152-token pool/four state slots;48,640 actual retrieval-input
  tokens at three depths, prefix-hit TTFT48.7–59.5ms; actual OpenCode harness
  passes and303.7s text soak completes without a recorded error.
- Maximum-context:81,920-token pool/one state slot;81,400 actual retrieval-input
  tokens at three depths; radix disabled, repeat-request TTFT14.20–14.39s;
  312.9s text soak completes without a recorded error.

Both complete64 thinking questions at62/64 versus63/64 on the same original
baseline IDs; reduced auxiliary results are32/32 greedy and PPL6.93278.
Their eight fixed greedy prompts (512-token cap) exactly match the retained
embedding control. These checks do not certify the0.5pp gate on new combinations.

## Retained full evidence

- Original baseline GSM8K:1285/1319,97.4223%, two capped responses.
- Converted embedding GSM8K:1281/1319,97.1190%, three capped responses;
  drop0.30326pp meets the recorded0.5pp full-run gate. Auxiliary:195/200,
  PPL7.86931963. These full results predate the final-combination short tests.
- A recorded vision configuration completes7,226.53s mixed runtime, restart and
  Windows contention checks, with one caption omission retained. It does not
  establish7200s stability of the two selected text configurations.
- Present-desktop memory scanning retains126 passing points across25
  combinations. These are bounded observations, not a physical/global maximum.

## Numerical and workflow findings

Six patches reconstruct eleven engine files exactly. Actual cached-KV replay
reproduces original native outputs; tile64→32 repairs the isolated attention
mismatch. Full-model NGRAM still matches0/8 strict outputs, so patch0006 remains
default-off and no lossless speculative recipe is promoted.

QAT screens62/64 thinking and32/32 greedy, with lower observed speed. FP32 state
screens63/64 thinking and31/32 greedy without a clear speed advantage. cuDNN and
FlashInfer GDN backend captures differ from the strict control; no backend is
adopted from a one-request speed result without full quality evidence.
Original-weight low effort screens64/64 with19,006 generated tokens versus
26,462 on the same retained baseline IDs (−28.18%). This separate sampled
protocol does not establish full quality or the embedding/low-effort combination.

The original-weight before/after speed controls drift about18%; candidate gains
cannot be inferred from the slower point alone. The original ≥1.5× lossless
speed target is unmet. A2× matched physical-context gain, Windows no-fallback
enforcement and the absent-desktop cohort remain unverified. Shared-memory
allocation counters do not establish CUDA spilling or explain the earlier slowdown.

The old long NVFP4-KV queue is cancelled with434 raw rows preserved:433 responses
plus one request error. Fresh official partial rescoring is418/433 versus424/433
on the same baseline IDs; this is not a full1319 score.
[Pre-delivery status archive](../results/status-history/20261112-status-before-budget-delivery.md).
