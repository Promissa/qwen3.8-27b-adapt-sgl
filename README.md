# Qwen3.8-27B on RTX 5090 D v2 (24 GB)

Measured local serving recipes and source adaptations for [PLAN.md](PLAN.md).
The [80%-context experiments](docs/profiling.md#实测上下文极限与-80-输入) compare maximum verified inputs, cold and cached latency, exact output tokens, LongBench v2 and BFCL. At 45,956 input plus 1,024 output tokens, the 48,128-token pool measured 1.845× end-to-end speedup with MTP.
The user-authorized three-hour, reduced-test delivery is complete; all10 planned
trials finished. [The delivery report](docs/three-hour-delivery.md) provides
completion time, raw evidence and limits.
The original speed and matched physical-context targets are not all achieved.

The two launchers use the converted embedding checkpoint already present in this
workspace. Set up the recorded runtime as described in [the recipe](docs/recipe.md),
then launch one server:

```bash
bash scripts/serve-balanced.sh
# Alternative for a larger one-off input; stop the previous server first.
bash scripts/serve-max-context.sh
```

Balanced has a49,152-token pool, verified48,640-token retrieval and49–59ms
prefix-hit TTFT. Maximum-context has an81,920-token pool and verified81,400-token
retrieval; it disables prefix reuse and repeats prefill in about14s. Both final
configurations complete reduced quality screening and pass API checks,8×512
greedy agreement with the retained embedding control and five-minute text stability. Actual
OpenCode integration also passes on balanced.

The separately audited full GSM8K results are1285/1319 for the original weights
and1281/1319 for converted embedding weights, a0.30326pp drop meeting the0.5pp
full-run gate. New64-question screening is62/64 for both final combinations
versus63/64 on the same original-baseline questions; it does not establish a
0.5pp bound. The earlier7,226.53s mixed vision runtime remains separate from the
short final-text tests.

Six patches exactly reconstruct eleven engine files from the recorded isolated
SGLang checkout. Patch0006 repairs the captured attention verify/decode mismatch
with tile32; full NGRAM strict agreement remains0/8, so speculative precision
patches stay off. The November 12 reduced delivery did not establish ≥1.5× lossless speedup or a matched physical 2× context gain. Windows no-fallback enforcement and the absent-desktop cohort remain
unverified. GPU tests run outside the restricted sandbox; the host GPU is usable.

See [current status](docs/status.md), [completion disposition](docs/completion-checklist.md)
and [measurement ledger](results/ledger.md). The superseded long NVFP4-KV run is
cancelled with434 raw records preserved. Earlier102 CPU checks and24 focused
checks for this revision are retained, alongside actual GPU replay and strict
negative results.
# qwen3.8-27b-adapt-sgl
