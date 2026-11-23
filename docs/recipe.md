# Measured 24GB recipes (Everyday PC)

**2026-11-21 Everyday-PC update:** This deployment targets an everyday 24 GB RTX 5090 D v2 PC where the Windows desktop stays on the GPU (typically 1–4 GiB variable occupancy). Absent-desktop and Windows Sysmem Fallback enforcement are formally dropped; all recommended launchers pin safe `--max-total-tokens` ceilings to stay clear of PCIe paging cliffs.

## Recommended 24GB launchers

All launchers use the NVFP4-embedding checkpoint (`weights/target-embedding-nvfp4`), FP8 E4M3 KV cache, BF16 recurrent GDN state, and single concurrency (`--max-running-requests 1`).

| Scenario | Launcher | Pool / Slots | Speculative / Backend | Verified performance (48K–256K) |
|---|---|---|---|---|
| **Default fast (everyday / thinking / coding)** | `bash scripts/serve-combined.sh` | 48,128 / 4 (lazy radix) | Quantized MTP (C1/D4) + deterministic Triton + `0007` graph verify + GDN prefill graph (`0008` off) | **150.2 / 157.7 tok/s** (1K/8K, **2.32×** e2e vs Ref), 1K/8K cold TTFT 86.5 / 729.2 ms, 47.6K warm TTFT 112–114 ms, BFCL 14/31 (313.8s, **1.64×** vs Ref), LongBench 5/16 |
| **No-speculative balanced** | `bash scripts/serve-balanced.sh` | 49,152 / 4 (lazy radix) | Off / FlashInfer | 63.1 / 62.6 tok/s, 8K cold TTFT 641–688 ms, 48.6K warm TTFT 47–50 ms (w/ GDN prefill graph), BFCL 18/31, LongBench 7/16 |
| **Long single-doc (VRAM-only)** | `bash scripts/serve-max-context.sh` | 81,920 / 1 (radix off; tested up to 155,300 at 0.8775) | Off / FlashInfer | 65.0 / 63.4 tok/s, 8K cold TTFT 631.7 ms, 81.4K 3-depth retrieval pass (14.7s repeat TTFT) |
| **Agent 256K (CPU offload)** | `bash scripts/serve-combined.sh --max-total-tokens 262144 --cpu-offload-gb 4` | 262,144 / 4 | Same as `serve-combined.sh` + 4 GiB weight offload | Verified 261,072-token synthetic probe + real 186K SWE-bench Pro Claude Code turns (2/3 Pro subset passed) |

> **Why `serve-combined.sh` keeps deterministic Triton + `0007`:** A 4-arm matched comparison (`results/detax-20261121/comparison.json`) proved that dropping `--enable-deterministic-inference` and switching to FlashInfer attention (766.7 ms 8K TTFT, 153.1 tok/s) or FlashInfer GDN prefill (828.9 ms 8K TTFT, 144.2 tok/s) is **slower** than deterministic Triton + `0007` (729.8 ms 8K TTFT, 157.2 tok/s) while losing bit-exact parity. `patch 0008` (full-attention MLP fusion, 0.999×) was removed from `serve-combined.sh`.

---

## Historical November 12 reduced-delivery launchers
Both use the converted embedding checkpoint, FP8 KV, BF16 recurrent state,
FlashInfer attention and one running request. Speculative patches remain off.
The [delivery report](three-hour-delivery.md) records the actual scope and limits;
these short tests do not certify every original PLAN target.

| Launcher | Static fraction | Token pool | State slots | Radix cache | Actual near-limit retrieval |
|---|---|---|---|---|---|
| `scripts/serve-balanced.sh` |0.88|49,152|4|Lazy prefix cache|48,640 input tokens, all three depths correct|
| `scripts/serve-max-context.sh` |0.92|81,920|1|Disabled|81,400 input tokens, all three depths correct|

Use balanced for repeated conversations and coding-agent work. Its measured
prefix-hit TTFT is49–59ms near the tested input limit. The maximum-context
variant fits a larger one-off document; repeated input is prefilled again,
with measured warm-request TTFT14.20–14.39s and no prefix hit.
Both pass API checks and complete64-question thinking screening (62/64, matched original
baseline63/64),18 PPL windows,32/32 nonthinking greedy questions and a300s text
soak. The actual OpenCode harness also passes on balanced. Both final8-prompt greedy (512-token cap)
captures match the retained embedding control exactly. The embedding checkpoint
already has a separately audited full1319-example result meeting the0.5pp gate;
64 examples cannot establish that bound for each new combination.

```bash
source scripts/runtime-env.sh
# Existing converted checkpoint is complete in this workspace.
bash scripts/serve-balanced.sh
# Stop the current server before launching the other profile.
bash scripts/serve-max-context.sh
# Inspect a launch without allocating GPU memory.
bash scripts/serve-balanced.sh --dry-run
# Reproduce the short original-weight control at24,576 pool tokens.
MODEL_PATH="$PWD/weights/target" bash scripts/serve-balanced.sh --max-total-tokens 24576
```

`MODEL_PATH` and trailing scalar arguments retain their overrides. Both launchers'
expanded effective parameters match their passed probes exactly. Their evidence
is `results/budget-3h-20261112/serve-*-command-validation-r2.json`.
The measurements below are historical source, protocol and candidate details;
current candidate dispositions are in the delivery report.

## Reduced low-effort observation

The original-weight24,576-token control completes64 thinking questions with
`reasoning_effort=low` at64/64, versus63/64 on the retained baseline's same IDs.
Generated tokens fall26,462→19,006 (28.18%). This is a single sampled subset
under a different request protocol. It does not establish full quality,
lossless speedup or the quality of low effort combined with the embedding
checkpoint. No default request effort changes. Default nonthinking/PPL/serving
requests in this trial are separate control observations.

## Environment and deployment

Run from the repository root on the WSL ext4 filesystem:

```bash
bash scripts/setup-env.sh
# Or reuse the existing main@c3c292b65a environment, without editing it:
bash scripts/setup-env.sh --reuse-local ~/projects/sglang/.venv
source scripts/runtime-env.sh
```

The live baseline needs one source fix: the Qwen3.8 checkpoint advertises
`Qwen3_5ForConditionalGeneration`, whose text-only implementation is rejected by
the unpatched CLI allowlist. Reproduce the isolated checkout without changing the
existing checkout:

```bash
mkdir -p engine
git clone --shared --no-checkout ~/projects/sglang engine/sglang
git -C engine/sglang checkout -b qwen35-language-model-only \
  c3c292b65ab42efd933e0c77cfb48eb69f1b81d0
# Reproduce the complete tracked source. Experimental flags default off.
for patch in "$PWD"/patches/000[1-6]-*.patch; do
  git -C engine/sglang apply "$patch"
done
source scripts/runtime-env.sh
```

Runtime setup selects this checkout through `PYTHONPATH` and records its base
commit and working-tree change. Regression checks preserve the other CLI guards.

The six-patch stack was independently applied to clean files from this exact
commit in a temporary directory. All eleven modified source files match the live
checkout byte-for-byte, with no uncovered tracked changes or untracked source
files (`results/budget-3h-20261112/patch-stack-validation.json`). This verifies source
reconstruction; numerical, quality and stability gates remain separate.

| Patch | Purpose | Tier / Current evidence |
|---|---|---|
|0001|Qwen text-only CLI allowlist|**Mainline (PR-ready)** · Text capacity/API/full baseline quality pass|
|0002|Pass quantization configuration/prefix to the embedding hook|**Mainline (PR-ready)** · Saves 1.64–1.77 GiB; full GSM8K 1281/1319 (-0.303pp, passes 0.5pp gate)|
|0003|Opt-in GDN verify beta casting and BF16 state persistence|**Mainline (MTP)** · Required for bit-exact GDN verify parity|
|0004|Opt-in Qwen MTP sharing before pool sizing|**Mainline (PR-ready)** · Fixes OOM during pool sizing at 0.92|
|0005|Opt-in tree-convolution accumulation order|**Mainline (MTP)** · Required for bit-exact tree conv1d verify parity|
|0006|Opt-in SM120 FP8-KV verify tile32|**Archived (default-off)** · Superseded by `0007`|
|0007|Opt-in C1/D4 in-graph decode-attention verify (`SGLANG_OPT_MHA_VERIFY_GRAPH`)|**Mainline (MTP)** · Restores 100% bit-exact CUDA Graph MTP (150–173 tok/s, 2.32×–3.06×)|
|0008|Opt-in full-attention MLP SiLU+FP4 fusion|**Archived (default-off)** · Measured 0.999× (no gain); removed from `serve-combined.sh`|
|0009|Share full packed NVFP4 embedding module with external Qwen MTP|**Mainline (PR-ready)** · Fixes load crash when combining NVFP4 embedding + MTP|

The experimental flags are `SGLANG_OPT_GDN_VERIFY_MATCH_PACKED`,
`SGLANG_OPT_QWEN_MTP_EARLY_SHARE`, `SGLANG_OPT_GDN_VERIFY_CONV_ORDER` and
`SGLANG_OPT_MHA_VERIFY_TILE32`.
All default off. The source findings do not promote a speculative recipe;
the paired deterministic full-model strict gates remain negative. The actual
eager control/NGRAM trace first differs at full-attention layer3 output with
matching Q/K/V inputs. Actual cached-KV replay now isolates that first difference
to the attention route/tile, while full-model strict equality remains failed.
[Actual findings](attention-verify-findings.md).

Default installation pins SGLang 0.5.21 as specified by the plan, plus sgl-eval
0.1.3. The reused environment is a different, explicitly recorded source baseline:
SGLang `0.0.0.dev21933+gc3c292b65`, torch 2.14.1, FlashInfer 0.7.3. Reuse mode does
not install missing packages. The live environment has the following evaluator
and proxy dependencies installed; reproduce them with:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  'sgl-eval==0.1.3' 'antlr4-python3-runtime==4.9.3' 'socksio==1.0.0'
```

The shared environment mixed CUDA 13.5 compiler support with 13.0 runtime headers.
The project overlays matching CUDA 13.0 packages without changing that environment:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-crt==13.0.88' \
  'nvidia-nvvm==13.0.88' 'nvidia-cuda-runtime==13.0.96'
source scripts/runtime-env.sh
.venv/bin/python scripts/fp4_smoke.py
```

`prepare_cuda.py` combines those local compiler/runtime files with the shared
cuBLAS headers/libraries in a workspace symlink tree, and supplies WSL's driver
link for JIT builds. Runtime setup also exposes the existing Ninja executable.
Help, FP4 imports and the real quantize/CUTLASS GEMM numerical check pass. First-use
GEMM build/test took 453.7 seconds; relative RMSE versus BF16 was .1344. Set
`MAX_JOBS=2` during JIT to limit host memory use. All generated caches stay in this
workspace. The shared SGLang source and unrelated tokenizer-manager change remain
untouched; both feasibility patches are confined to the isolated checkout.

Before measuring memory, confirm NVIDIA Control Panel's CUDA Sysmem Fallback Policy
is **Prefer No Sysmem Fallback**. Move display output to the integrated GPU, close
other GPU applications, and check `.wslconfig` memory allowance. These Windows
settings need explicit verification. The NVAPI query works, but the requested
policy write returned SETTING_NOT_FOUND (-160); no change was saved.
`results/windows-fallback-policy-applied.err` retains the rejection. A successful command in
the user's terminal does not establish device access in a sandboxed agent session.
The direct private-ID test in `results/windows-fallback-private-unsaved.json`
also returns−160 for SetSetting. It destroys the temporary session without
SaveSettings and confirms unchanged fresh-session reads; policy remains unverified.

The post-restart default read-only query still returns−160 for this setting
and both profiles; its public power-management control resolves normally.
[Fresh NVAPI evidence](../results/windows-fallback-read-20261112.json).
NVIDIA's [published procedure](https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion)
selects a Windows Python executable and requires restarting it. The reviewed
[WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) does not
document how this policy targets/enforces a WSL workload. Neither setting-not-found
nor the missing instructions prove a default value or non-support. Actual WSL
policy scope and a matching fresh-process/residency cohort remain to be verified;
the current process/settings were not changed.
[Evidence review](../results/windows-fallback-documentation-review-20261112.json).

The cancelled NVFP4-KV quality cohort retains read-only Windows memory samples.
DXGI maps its counter LUID to the5090 D v2; adapter-level usage reports23.434GiB
dedicated and4.151GiB shared. Some per-process counters violate size/commit
bounds, so they cannot be summed as physical usage. A vmwp shared-memory row
does not identify the Linux workload or CUDA spilling. This remains an
uncertified memory cohort with an unverified fallback policy.
[Counter/mapping analysis](../results/progress-checks/kv-nvfp4-windows-memory-analysis-20261112.json).

```bash
bash scripts/download.sh target
.venv/bin/python scripts/quantize_embedding.py weights/target weights/target-embedding-nvfp4
bash scripts/serve-balanced.sh
```

The downloader resolves and records an immutable revision before downloading,
reuses completed files from that revision, and marks completion only after all
selected files arrive. Installed Hub1.33.0 resumes HTTP offsets inside the same
process, but uses process-unique partial files and removes them on failure;
restarting a process does not resume those byte offsets. The timing guards pause
and resume the same verified downloader instead. QAT's official/mirror1MiB range
probes returned identical bytes from the same CDN, so no host switch was justified
(`results/qat-http-range-probes.json`).
Launch scripts use local checkpoints and reject incomplete download manifests.
`MODEL_PATH`, `DRAFT_MODEL_PATH`, `HOST`, and `PORT` can override the defaults.

## Baseline feasibility and memory envelope

```bash
bash scripts/serve-balanced.sh --dry-run
bash scripts/probe.sh --profile balanced --desktop absent \
  --output results/probes/first-baseline
# Diagnostic fallbacks, each with a new evidence directory:
bash scripts/probe.sh --profile max-context --desktop absent \
  --output results/probes/no-radix
bash scripts/probe.sh --profile balanced --desktop absent \
  --output results/probes/vision -- --with-vision --chunked-prefill-size 1024
```

The probe owns and terminates its server process group. Passing requires READY,
a 95%-of-reported-capacity prefill, and exact 8192 input / 1024 output generation.
Concurrency 2 means two actual simultaneous 8192/1024 requests. Random IDs are
capacity input only; acceptance and quality must use real prompts.

`probe.json` retains the command, environment, checkpoint revision when available,
server info before/after requests, parsed pool sizes, full relevant log lines,
request lengths and timings, errors, and sampled device memory. `server.log` and
`gpu.json` are raw supporting evidence. Device memory is sampled at 1s and includes
other GPU users; it is not an allocator peak. Missing log fields remain unknown.
New probes also retain `runtime_provenance` and `engine.patch`, identifying the
actual imported code, dependency versions, commit, working-tree diff and probe
hash. Converted checkpoints include their conversion metadata. The older setup
environment snapshot is retained separately from this launch-specific evidence.

```bash
.venv/bin/python bench/envelope_sweep.py --desktop absent
# After concurrency 1 succeeds:
.venv/bin/python bench/envelope_sweep.py --desktop absent --concurrency 2
# Run separately with actual desktop GPU occupancy restored:
.venv/bin/python bench/envelope_sweep.py --desktop present
```

The grid covers four cache strategies × three prefill sizes, plus one fp32
control. A coarse scan discovers a passing interval before bisection; lower
fractions can also fail to allocate pools. The result is the highest **tested**
passing fraction at the requested precision, not proof of a global maximum.
Infrastructure failures stop the sweep and cannot become negative capacity data.
The desktop label is a manually observed condition, not a simulated allocation.

| Recipe | Weights revision | Context / KV tokens | State slots | Peak MiB | 8192/1024 |
|---|---|---|---|---|---|
| balanced text candidate, desktop present, fraction .88, chunk2048 | `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33` | Pool36,512; tested input34,685 | 11 logical | 22,584 device-wide | Passed,15.80s |
| text pin4 control, fraction .88, chunk2048 | Same target | Pool52,522; tested input49,894 | 4 pinned | 23,055 device-wide | Passed,15.80s |
| max-context candidate, desktop present, fraction .88, chunk2048, radix disabled | Same target | Pool62,109; tested input59,002 | 1 automatic | 22,697 device-wide | Passed,16.37s |
| vision candidate, desktop present, fraction .88, chunk1024 | Same target | Pool15,168; tested input14,408 | 4 automatic | 22,920 device-wide | Passed,15.82s |

These are individual passing workloads, not maxima or final stable operating
envelopes. Raw evidence is in `results/probes/baseline-near-limit-088/`,
`results/probes/pinned-state-4/` and `results/probes/vision-first/`. Their complete
probes pass; state slots are logical and peaks are sampled whole-device usage.
The radix-disabled capacity point is
`results/probes/2026-11-10T072544.182031_0000/present-disabled-2048-c1-bfloat16-0.88000/`;
its strict greedy, natural retrieval and full quality gates remain open.
Pool sizes and desktop occupancy vary across restarts. Windows sysmem fallback
remains unverified. On the2026-11-11 resumed session, prelaunch GPU use
is3299MiB: vision.88/chunk1024 fails startup because only3 hybrid state slots fit.
The new.92/pin4/24576 configuration passes capacity,8/8 own-control tokens,
API and near-limit retrieval. The two-hour mixed run completed in 7,226.53s
with 913 successful responses and zero observed OOM log events. Frame26 omitted
the blue circle. The 24,064-token post-run retrieval passed at all three depths,
with one warm TTFT of 1.853s. [Validated runtime report](../results/soak-snapshots/vision-runtime-observation-r2-092-pin4-final/summary.json).
These are different occupancy cohorts. In the later no-buffer scan,.96/.98 capacity points pass but
fixed8192/1024 requests slow to90.22/85.96s with large PCIe traffic. A successful
high-fraction capacity request alone cannot choose the balanced recipe; see
`results/envelope-pcie-phase-observation.json` for the scoped observation.

The present-desktop scan's first validated table snapshot is
`results/envelope-snapshots/20261110-1000/table.md` (95 points,15/25 searches
completed at capture). Its JSON retains each probe hash, source diff, timings
and independently checked active-request telemetry. Generate a new snapshot
without launching a server:

```bash
.venv/bin/python bench/summarize_envelope.py \
  --csv results/envelope-present-complete.csv \
  --output results/envelope-snapshots/next-snapshot
```

Existing snapshots are preserved. Highest tested passing fraction is not a
proven maximum, and a capacity probe's request time is not a formal repeated
benchmark. The table keeps incomplete combinations pending.

The sweep now tests low/high endpoints first. A passing high endpoint settles
the same bounded search; if it fails, coarse discovery still finds a passing
middle interval before bisection. `--resume` reuses retained CSV measurements
and completed summaries, rejects protocol/infrastructure mismatches, and never
reruns cached samples. The clean upgrade preserves all114 earlier measurements;
see `results/envelope-endpoint-handoff.json`. All25 combinations remain required.

Record actual pre/post-load free memory and runtime reserve from logs. The fixed
source defaults `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` to false. Its 3712 MiB
activation reserve for TP=PP=1, chunk≤2048 only matters when its guarded branch is
active; do not subtract it from every budget. See [source-audit.md](source-audit.md).

## Speculation and ablations

```bash
bash scripts/serve-ngram.sh
bash scripts/download.sh dspark
bash scripts/serve-dspark.sh
# Only after separately quantizing and validating a DFlash draft:
DRAFT_MODEL_PATH=/absolute/path/to/quantized-dflash bash scripts/serve-dflash.sh
# Only with an MTP-bearing reference checkpoint:
MODEL_PATH="$PWD/weights/reference" bash scripts/serve-eagle.sh
```

DSpark pins eight Mamba slots for concurrency 1; the probe uses `4*C+4` for higher
concurrency. EAGLE refuses the default target because it has no MTP head. DFlash's separate CPU converter is available:

```bash
.venv/bin/python scripts/quantize_dflash.py weights/dflash weights/dflash-nvfp4
```

It quantizes attention/MLP linears to W4A16_NVFP4, preserving BF16 activations,
context projector, convolutions and selector. It uses a common global scale for
each fused QKV and gate/up group. The pinned real draft shrank from 3.85GB to
1.55GB. A fixture loads through the actual SGLang fused loader; GPU load and
acceptance are still unverified. This candidate needs no activation calibration.
The existing upstream A16 method uses Marlin; backend compatibility is a live gate. Failed speculative launches must be recorded,
including measured shortfalls where observable; no speedup is assumed.

```bash
# Capture commands without starting a server:
.venv/bin/python bench/run_experiment.py baseline --tag baseline-r1 \
  --desktop absent --quality --dry-run
# Live: capacity gate, real-token capture, API checks, 2 loads × 3 repetitions;
# --quality adds GSM8K 1319, EN/DA/code perplexity and the old greedy GSM8K 200 check.
# --hard-quality gpqa selects GPQA Diamond 198 if the account has dataset access.
.venv/bin/python bench/run_experiment.py baseline --tag baseline-r1 \
  --desktop absent --quality
.venv/bin/python bench/run_experiment.py skip-decode-lock --tag skip-lock-r1 \
  --desktop absent --quality
```

Each experiment in `configs/experiments.json` starts from its named control, with
other catalogued environment switches cleared. Adjust one flag or optimization
at a time. API/quality outputs alone do not promote a recipe: compare the stored
token IDs against the control and fill the ledger. Speculative parameter sweeps,
QAT/reference loading, lossless-vs-lossy combination studies, and final promotion
remain live experimental work.

## Quality, speed, and profiling

```bash
.venv/bin/python bench/eval_quality.py --tag balanced --eval gsm8k
.venv/bin/python bench/eval_quality.py --tag balanced --eval gpqa
# Separate effort comparison, with the same weights and output cap:
.venv/bin/python bench/eval_quality.py --tag balanced-medium --reasoning-effort medium
.venv/bin/python bench/quality_battery.py --output results/quality/battery-balanced.json
.venv/bin/python bench/compare_battery.py \
  --baseline results/quality/baseline-logprob-chunk128/battery.json \
  --candidate results/quality/battery-balanced.json \
  --output results/quality/battery-balanced-comparison.json
.venv/bin/python bench/long_context.py --input-tokens 8192 \
  --output results/quality/retrieval-balanced-8k.json
.venv/bin/python bench/api_smoke.py --output results/quality/api-balanced.json
# With a vision-enabled server and an actual PNG fixture:
.venv/bin/python bench/api_smoke.py --image /path/to/fixture.png \
  --output results/quality/api-vision.json

# Capture on baseline, restart server with candidate, compare exact output IDs:
.venv/bin/python bench/token_compare.py --output results/quality/baseline-tokens.json
.venv/bin/python bench/token_compare.py --output results/quality/candidate-tokens.json \
  --reference results/quality/baseline-tokens.json

bash bench/run_serving_bench.sh balanced
PORT=30000 .venv/bin/python bench/seat_ttft.py balanced 8192 3
.venv/bin/python bench/capture_profile.py --output results/profile/balanced
# With --enable-metrics on the server; retain all repetitions:
.venv/bin/python bench/conc_ladder.py --n 1 2 --reps 3 \
  --output results/bench/balanced-real-ladder.json
```

GSM8K sampling is thinking enabled, temperature 1.0, top_p .95, top_k 20, 1319
examples, one repeat, seed 1701. Results retain predictions, errors, and truncation metadata;
do not interpret a partial evaluation as a score. `--max-tokens` defaults to16384,
matching the completed control; `run_experiment.py --quality` records that cap explicitly
and accepts `--quality-max-tokens` for a separately matched protocol. The cookbook's
32k protocol and a50k single-case follow-up are separate comparisons. Choose a
capacity-tested cap and keep it identical across controls/candidates.
Native effort levels are `low`, `medium`, and `xhigh` (the default). They change
instructions in the chat template. They do not set an output-token budget.
All1319 rendered math prompts match between target, reference and QAT tokenizers
in each effort mode (`results/gsm8k-template-parity.json`); that check does not
establish equivalence for tool histories or multimodal prompts.
After both full runs complete, compare their UUID result directories:

```bash
.venv/bin/python bench/compare_quality.py \
  --baseline-dir results/gsm8k/baseline-full-16k/sgl_eval_gsm8k_CONTROL_UUID \
  --candidate-dir results/gsm8k/CANDIDATE/sgl_eval_gsm8k_CANDIDATE_UUID \
  --output results/quality/comparison-CANDIDATE.json
```

This rejects partial runs, sampling mismatches and silently clamped budgets. It
checks official row scores against the aggregate and records lost/gained answers.
At1319 questions, six net lost answers fit the0.5pp limit; seven exceed it. A
passing GSM8K comparison alone does not promote a recipe.
The16k control scores1285/1319 (97.4223%), with two truncations; validated
evidence is `results/quality/baseline-full-16k-gsm8k-validated.json`. This is a
different output cap from the cookbook32k comparison. Later auxiliary PPL
OOMed; recovery passes all120 PPL windows and192/200 greedy GSM questions
(96%, seven responses hit the768-token cap) using `SGLANG_ENABLE_LOGPROB_CHUNK=1` and
`SGLANG_LOGPROB_CHUNK_SIZE=128` without changing model weights.
Lossy adoption requires ≤0.5pp degradation,
GPQA/quality battery and long-context retrieval, not only a short smoke test.

The speed driver uses the cookbook `sglang-oai` backend, random input/output
lengths 1024/1024 and 8192/1024, range ratio 1, flush cache, concurrency 1, three
repetitions, ten requests per repetition by default (`NUM_PROMPTS` overrides).
Set `MODEL_PATH` to the running checkpoint when it differs from `weights/target`;
the driver uses that local tokenizer and sends the served model name separately.
Inspect failures and actual lengths before aggregating TTFT, TPOT, throughput and
acceptance. Real prompts are reused from the old project. Raw synthetic throughput
is not evidence of useful speculative acceptance.
`seat_ttft.py` labels its first sample `first_in_series`; it cannot establish a
cold server. Startup-to-READY time comes from the owned probe's `ready_s`.
The probe's first external generation is its near-capacity, one-output-token
request, whose recorded `elapsed_s` is end-to-end time rather than streamed TTFT.

The profiler records CPU/GPU/MEM by stage, with stack/shapes. Before engine
patches, attribute decode time to GEMM, GDN, attention, lm_head, sampler, draft and
scheduling; inspect trace overlap before summing durations. No patch is justified
by an unmeasured estimate.

## Embedding optimization

```bash
.venv/bin/python scripts/quantize_embedding.py weights/target weights/target-embedding-nvfp4
MODEL_PATH="$PWD/weights/target-embedding-nvfp4" bash scripts/serve-balanced.sh
```

The CPU converter packs E2M1 pairs, E4M3 scales per 16 values, and an FP32 tensor
scale. It updates mixed-precision metadata so the existing SGLang embedding
method is selected. It rewrites only the affected shard and symlinks unchanged
shards, preserving the source checkpoint. Keep the source files available.
Tied embeddings and ambiguous exclusions are rejected. Its synthetic layout
tests exercise the actual upstream gather method. This proves layout compatibility
on fixtures. The actual converted checkpoint now passes load/capacity/API and
84,787-token-budget retrieval after patch 0002 forwards quantization config and
prefix through the backbone embedding hook. Its pool was 85,299 tokens in that
run. Full GSM8K at a24,576-token pool cap now scores1,281/1,319
(97.1190%, three truncations, zero request errors). Independent official
rescoring agrees on every label and extraction. The0.30326-percentage-point
drop from the initial target passes the0.5-point GSM8K gate. Auxiliary quality
completed all120 PPL windows and200 greedy examples:195/200 correct, six capped
responses and overall PPL7.86932 (baseline192/200 and7.89022). Raw inputs,
logprobs, answers and aggregates independently agree in the pinned runtime.
[Full quality evidence](status.md#retained-full-evidence).
The selected combinations now have the reduced screening described above. Full
final-profile certification and sysmem policy verification remain open; this older
85k observation is not a certified physical maximum.

## Multi-round prefix-policy comparison

`bench/prefix_dialogue.py` starts an8k archive with six different keys spread
across its depth. Each subsequent question asks for a new key. It renders the
actual growing message history with the local tokenizer, preserving the model's
answers, and retains exact input IDs. After the complete conversation, each of
those same inputs runs again after a cache flush. Three repetitions retain all
warm and flushed timings and responses, rather than selecting the fastest one.

```bash
# Against an idle owned server; run once per separately booted cache policy:
.venv/bin/python bench/prefix_dialogue.py --input-tokens 8192 \
  --rounds 6 --reps 3 --output results/quality/prefix-dialogue-lazy.json
```

The initial policy pair changes `--mamba-radix-cache-strategy` between
`extra_buffer_lazy` and `no_buffer`; both use the original target, four pinned
state slots and a24,576-token pool cap. Capacity, eight-prompt token equality,
multi-round comparison and near-limit retrieval run in owned servers. The
download is excluded from these timed trials. Queue evidence is
`results/prefix-policy-probes.json`. Lazy passes all36 dialogue/control requests:
follow-up median TTFT48.2ms/max246.8ms, matched flushed median673.9ms.
No-buffer was rejected before launch: it requires
`--disable-overlap-schedule`. Its retry must retain that required difference;
the corrected retry now passes all36 dialogue/control requests, baseline
8-prompt equality and near-limit retrieval. No-buffer/disabled-overlap follow-up
median47.9ms/max228.3ms, flushed median666.5ms. This is close to lazy and does
not establish a speed advantage. Both trials pin four state slots, so they do
not measure automatic-policy state-slot savings.

Passing the dialogue requires correct retrieval, exact warm/flushed output IDs,
zero cache counts for flushed controls and initial turns, actual prefix hits
on every follow-up, and every follow-up TTFT≤1.5s. Offline construction verifies
the canonical histories share more than8k tokens; four evidence tests reject
missing hits, unflushed controls, token differences and slow follow-ups. None of
those offline checks establishes runtime cache reuse or general agent quality.

## External MTP candidate

The reference MTP's seven attention/MLP matrices have been converted to
W4A16 NVFP4, leaving its ordinary context projector and norms BF16. Actual MTP
payload drops from849,398,784 to314,297,372 bytes (535,101,412 saved, about0.498GiB).
The backbone is unchanged: two shards remain source symlinks, all55 unmodified
tensors in the rewritten shard compare byte-for-byte, and all401 original
quantized-layer entries are retained. The real Qwen MTP constructor and fused
weight loader pass a converted fixture. This is an offline result.

```bash
# Use a new destination; keep the reference source checkpoint available.
.venv/bin/python scripts/quantize_mtp.py weights/reference weights/reference-mtp-nvfp4
# Experimental command: first trial fails target pool sizing at.92:
DRAFT_MODEL_PATH="$PWD/weights/reference-mtp-nvfp4" bash scripts/serve-eagle.sh \
  --chunked-prefill-size 1024 --max-mamba-cache-size 4 \
  --max-total-tokens 24576 --mem-fraction-static 0.92
```

The main target remains the all-linear checkpoint. An explicit independent draft
path supplies the MTP absent from that target; its config is rewritten to the
Qwen MTP architecture by the existing engine. Static geometry matches, but
the first GPU trial loads target17.46GiB plus draft2.93GiB and fails before
READY during target-pool sizing. The unmodified lifecycle shares embedding/head
after that allocation. Default-off patch0004 moves the existing sharing earlier
and passes24576-token capacity at the same.92 point, but strict greedy is0/8.
Losslessness, acceptance and speed remain unverified; see
`results/external-mtp-probe.json` and `results/eagle-early-share-queue.json`. No new quantization kernel
is introduced. Shared CPU packing also retains the original DFlash fixture
result; the exact historical DFlash converter is archived under
`results/converter-sources/` with its original recorded hash.

## Stability and final delivery

```bash
.venv/bin/python bench/soak.py --output results/soak/balanced-text
# Mixed vision/tool/long/short workload for PLAN.md stage 6:
.venv/bin/python bench/soak.py --image bench/fixtures/vision-shapes.png \
  --uncached-vision-fixtures --continue-vision-errors \
  --output results/soak/final-mixed
```

PLAN.md stages 4 and 6 require an image input smoke check and two hours of
mixed traffic without an observed OOM. The independent report's
`runtime_stability_7200s_validated` supplies the runtime result for the mixed
command above. Check `vision_included` and the attempted case counts for workload
coverage. The initial text-only command exercises the deployment mode in §9.

The collector retains failed captions in `vision_quality_errors`. The historical
`full_plan_soak_gate` field and the report's `full_mixed_7200s_validated` track
the additional check that every mixed response meets its expected content.
`--continue-vision-errors` keeps collecting runtime evidence after a caption
failure. Transport, request and monitor failures stop the collector.

The caption request explicitly asks for words without
bounding boxes; an old Frame38 request format produces coordinates in a fresh
process. Color/shape checks allow punctuation between adjacent expected words,
but still reject incorrect associations. Raw failed captions and image digests
are retained. The regression suite checks punctuation, incorrect pairs and early
telemetry failure. The unique-frame mode redraws native test geometry with a
new frame number per image request, so decoded-image/embedding cache reuse does
not hide repeated encoding. Fixture responses must identify the red square and
blue circle. Every workload case must complete. Telemetry includes device usage
and `/proc/meminfo`; valid samples must cover the requested duration, including
both ends, with no gap above10s. Missing coverage or monitor errors fail the gate.
Frozen raw snapshots and PNG/PDF memory/latency curves can be generated without
modifying the serving environment:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  uv run --no-project --python .venv/bin/python --with matplotlib==3.10.8 \
  python bench/report_soak.py results/soak/RUN --output results/soak-snapshots/NEW_SNAPSHOT
```

The report rechecks actual long-request counts, tool arguments, reasoning
separation, vision object text and unique-frame hashes, then recomputes the
terminal telemetry/duration gate. Running snapshots explicitly remain incomplete.

The .92 vision configuration passed its fresh restart with 24,576 tokens of
capacity, 8/8 control outputs, API and three-depth retrieval. A separate Windows
contention run used the .88 text baseline with 28,139 tokens of capacity and
512MiB of nominal native D3D11 resources. All nine requests finished and strict
tokens, API and retrieval recovered after release.
[Restart and Windows results](status.md).
Each export freezes both raw files and the report source, with SHA256 hashes.
The completed .92/pin4/24,576-token mixed run lasted7,226.53s and retained6,826
valid telemetry samples,913 successful responses and114 unique images. One
caption omitted the blue circle; its response and image digest remain in the
report. The runtime gate passes with zero observed OOM events; the additional
all-responses-content gate fails.
[Frozen report and curves](../results/soak-snapshots/vision-runtime-observation-r2-092-pin4-final/summary.json).

Restart, Windows contention and actual OpenCode model/tool integration have
completed for their recorded configurations. The absent-desktop cohort, fallback
policy verification and full final-combination qualification remain open. The
selected named launchers have the reduced validation described above; the earlier
two-hour vision result does not certify their two-hour stability.

## OpenCode V2 integration fixture

The installed OpenCode is v2.2.6. Its current provider and permission schemas
differ from the cookbook's V1 example; see the official
[provider documentation](https://opencode.ai/docs/providers/) and
[permissions documentation](https://opencode.ai/docs/permissions/).
`configs/opencode-v2.json` passed isolated configuration preflight.

```bash
# Against an idle owned server, after performance measurements:
.venv/bin/python bench/run_harness.py --output results/harness/balanced
```

The fixture uses local SGLang, project-local OpenCode state and a private server.
It permits reading/editing the fixture and the exact unittest command. Raw events
and an independent test result are retained. The actual model completed read,
edit and shell tool calls, changed the fixture source and passed unchanged
independent tests. [Completed fixture](../results/harness/baseline-harness-pwd-recovery/harness.json).

## Experimental MTP lifecycle patch

`patches/0004-qwen-mtp-share-before-pool.patch` adds the default-off
`SGLANG_OPT_QWEN_MTP_EARLY_SHARE=1`. It applies the existing embedding/head
sharing contract before target-pool sizing and rejects unsupported startup,
parallelism, token maps and packed embeddings. A converted-Qwen CPU fixture
checks whole NVFP4-head sharing, scales and unchanged draft parameters.
The retained early-sharing trial passes actual24576-token capacity at.92 where
the late-sharing control failed, but strictgreedy remains0/8. No lossless or
faster speculative recipe is promoted.

The unauthenticated localhost provider needs no credential. OpenCode run uses
inherited PWD to change directory, overriding subprocess cwd; `run_harness.py`
sets PWD to its fixture so its local model config and tools use that location.
The corrected actual model/tool fixture passes: read/edit/shell tools complete,
the assistant responds after tools, source changes, tests remain unchanged and
independent fixture tests pass. Evidence:
`results/harness/baseline-harness-pwd-recovery/harness.json`. See
[custom providers](https://opencode.ai/docs/providers) and model
[location scope](https://opencode.ai/docs/models).

## Full-quality capacity preflight

`bench/eval_quality.py` now renders all1319 actual upstream GSM8K prompts with
the active local tokenizer before starting evaluation. At the default16k output
cap, the current longest prompt is263 tokens, requiring16647 tokens total.
The real-data boundary checks reject16646 and accept16647
(`results/quality-capacity-preflight-boundaries/`). The driver never reduces
the cap to accommodate a small pool. Effort-specific prompts are counted again.

`bench/resume_quality.py` requires the same checkpoint/revision, serving
source/runtime, complete generation protocol, all effective serving settings
and experimental environment switches before it generates missing responses.
Only restart timings/internal state/default server seed are excluded from the
serving-settings comparison, and an integer per-request seed is mandatory.
Changed KV/state precision, backends, graph settings or speculative options
are rejected. A read-only validation of the completed embedding resume matches
all561 compared serving fields and5 environment fields; the original helper
and raw responses remain preserved.
[Retained identity check](../results/quality/embedding-resume-serving-identity-20261112.json).

After all1,319 predictions and final metrics exist, independent official
rescoring uses the saved generation text and upstream gold answers. It checks
both answer extraction and scores; stored labels are never given to the scorer.
Use a fresh output path to preserve prior evidence:

```bash
.venv/bin/python bench/rescore_quality.py \
  --run-dir results/gsm8k/embedding-full-resumed-20261112/sgl_eval_gsm8k_resumed \
  --output results/quality/embedding-full-resumed-20261112-official-rescore.json
```

Reference and metadata-only mixed-control rescoring each reproduce all1,319
labels/extractions. A passing consistency report does not override the separate
0.5-percentage-point quality gate or qualify a final recipe.

For a completed candidate, the combined independent audit uses a fresh output
directory and performs no generation:

```bash
.venv/bin/python bench/audit_quality_run.py --tag CANDIDATE_TAG \
  --output results/quality/audits/CANDIDATE_TAG-r1
```

It requires complete1319/16k records, re-extracts and officially rescores all
answers, validates the report's raw hashes and individual rows, and recomputes
the matched auxiliary comparison. `completed_audit` describes evidence
validation; its GSM8K gate can still be false. `--separate-low-effort` accepts
only the declared low-effort protocol difference and reports it separately,
without a matched weight-quality or auxiliary claim. The future numerical
queue runs these audits after its GPU jobs, retaining any incomplete/failed
evidence. The existing embedding audit is
[recorded here](../results/quality/audits/embedding-full-resumed-20261112-r1/audit.json).

`eagle-native` is a separate experimental profile without ReplaySSM; the existing
`eagle` profile retains its prior default. Native/packed-GDN controls and higher
fraction DFlash retries remain gated experiments, not deployment recipes.


## Windows application occupancy and recovery

`bench/windows_contention.py` runs against an already owned model server:

```bash
source scripts/runtime-env.sh
python bench/windows_contention.py --mib 512 --output results/contention/NEW_TAG
```

Use it as a `bench/probe.py` validation step so model launches remain serial.
The native PowerShell fixture creates its own NVIDIA D3D11 textures, waits for
GPU copies, and releases them through a stop file and `finally`. Nominal resource
size is512MiB; physical residency remains an observation from separate telemetry.
The benchmark retains three flushed8192/1024 requests before, during and after,
plus API recovery. The driver also checks strict original tokens before
and after and natural near-limit retrieval. It does not change display outputs,
GPU policy, or existing applications. Microsoft SDK signatures/vtable indices
are retained in `results/windows-d3d11-sdk-abi.json`; native compile/ABI validation
passes without GPU calls. The actual run completed all nine requests, recovered
the original eight-prompt token control and passed API and near-limit retrieval.
[Independent actual-run validation](../results/contention/baseline-windows-contention-r1/independent-validation.json).
This cannot substitute for absent-desktop measurements or
verification of the CUDA fallback policy.

## Numerical backend comparisons and historical full protocol

The QAT checkpoint has an additional CPU-only loading preflight. It checks all
496 quantized text projection geometries, all176 fused global-scale groups,
and four representative real payload groups through the native Qwen and
compressed-tensors loaders and CUTLASS-path CPU preparation. All16 loaded
parameter digests match independently streamed checkpoint byte ranges; tiled
scale roundtrips and FP32 alpha checks pass. QAT also quantizes the96 GDN b/a
projections, unlike the initial target. The preflight stubs capability/backend
selection and performs no GPU GEMM. Its bounded GPU trial and reduced-quality
disposition are recorded in the delivery report; no full-quality result is inferred. [Report](../results/qat-loader-preflight-20261112-r2.json),
[independent checks](../results/qat-loader-preflight-20261112-independent-validation.json).

To repeat the bounded CPU check with a fresh output:

```bash
source scripts/runtime-env.sh
.venv/bin/python results/driver-sources/check-qat-loader-20261112.py \
  --output /tmp/qat-loader-preflight.json
```

The earlier cuDNN FP4 and FlashInfer GDN decode/prefill experiments pass
capacity but fail8-prompt strict equality. The current bounded trials repeat
API/token checks and two single-request loads under the user-authorized reduced
protocol. Their actual outcomes are in the delivery report. The old resume and
numerical-followup handoffs are superseded; they are not active future jobs.

The original three-repeat prioritization protocol required at least2percent
mean gain in both loads, all candidate repetitions above the control range,
and then full1319-example quality, auxiliary, retrieval and stability checks.
Those full requirements are not inferred from the reduced runs. The one-request
screening figures do not establish significance or support adoption of a backend
whose strict/quality evidence is incomplete.

This does not waive losslessness for speculative decoding, sampling or cache
changes. The FP32 recurrent-state comparison completes64 thinking questions
at63/64, matching the same original-weight baseline subset. Its reduced battery
is a coarse control, not a full-quality reference or a measurement of a small
BF16 state loss.

For NVFP4 KV with FlashInfer prefill and TRT-LLM MHA decode, include the shared
FP8 prefill workspace in memory arithmetic. This Qwen geometry uses18,432 bytes
per token for packed KV/scales and2,048 for the one cross-layer workspace,
giving the native20,480-byte cell size. At equal KV byte budgets the nominal
ratio to FP8's32,768 bytes is1.6;1.78 counts only persistent packed data/scales.
The native constructor was checked on meta tensors at the actual75,520-token
pool plus64 padding rows: total buffers1,547,960,320 bytes, shared workspace
147.625MiB. This is source/storage arithmetic, not measured allocator residency
or an explanation of the observed slowdown.
[Accounting](../results/memory/kv-nvfp4-full-storage-accounting-20261112.json).
