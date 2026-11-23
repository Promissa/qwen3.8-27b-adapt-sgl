# Optimization ledger

The November 19 [80%-context matrix](context80-20261119/comparison.json) adds 45,956, 82,262 and 124,240-token inputs. The largest verified single input was 155,300 tokens with one generated token. The 48,128-pool MTP pair measured 1.845× end-to-end on 45,956/1,024, with 2/2 exact outputs. GDN reduced warm TTFT from 64.82 to 56.84 ms at 82,262. See [methods and paired application scores](../docs/profiling.md#实测上下文极限与-80-输入).

The six baseline speed repeats pass: 1024/1024 mean 69.449 output tok/s,
8192/1024 mean 65.389. A clean trace and near-limit retrieval pass. Skip decode
lock matches all eight greedy prompts and passes every validation step, but
shows no speed gain. Full baseline GSM8K is1285/1319 (97.4223%, two truncations);
the complete auxiliary battery is192/200 greedy (seven capped responses) plus
120 PPL windows. Candidate quality and final optimized combinations remain
unmeasured. No cumulative gain is claimed.

| Item | Current evidence / decision | Required next evidence |
|---|---|---|
| Baseline feasibility | READY, 34,128/1 and 8192/1024 pass; 35,926 pool tokens in first passing point | Full memory grid, vision and final named recipes |
| Qwen3.5 text-only CLI fix | One architecture added in isolated source; 17 focused tests pass; real model passes capacity and text checks | Broader upstream review/regression scope if submitted |
| Runtime reserve | Post-capture switch defaults false; clean baseline confirms it stayed off. Padding slot already charged; text-only multimodal reservation0 | Retain actual branch/log evidence for each new configuration |
| Cache strategy / chunk / concurrency | Present-desktop grid runs live; all completed points pass capacity so far, including demonstrated C2 activity | Finish all25 combinations; absent-desktop/no-fallback verification; distinguish slow passes from balanced recipe choices |
| Skip decode lock | All gates pass, exact token equality; 69.068 / 65.373 tok/s (−0.55% / −0.02%); no speed gain | Matched restart controls before attributing memory-pool differences |
| Host-only checkpoint / hicache | All gates pass, 68.731 / 64.526 tok/s (−1.03% / −1.32%); 35,896 pool, 35,384 retrieval budget; observed process-group PSS 7.68GiB, memlock failed | No speed adoption; still allocates 11 GPU state slots, so pinning is separate |
| Graph pool precarve | All gates pass; 69.250 / 65.359 tok/s, no gain over initial control; pool 33,043 at this restart | No promotion; compare memory at matched desktop occupancy |
| GDN FlashInfer decode / prefill | Both start and pass capacity on SM120; both diverge on all eight greedy prompts (first differences at tokens 1–78) | Reject as lossless switches; recurrence is only ~2.2% of baseline summed GPU duration, so no lossy adoption is justified |
| CUTLASS / cuDNN FP4 | CUTLASS all gates pass, 68.980 / 65.326 tok/s; cuDNN capacity passes but 1/8 greedy prompts diverges at token 71 | cuDNN not admitted as a lossless optimization; no quality or speed gain claimed |
| FP32 state | Initial capacity/API/retrieval pass,32689-token pool | Full quality and speed; one full-grid feasibility control |
| NVFP4 KV | Initial capacity/API/retrieval pass,52096-token pool using FlashInfer prefill +TRTLLM-MHA decode | Full matched GSM8K/auxiliary quality and speed; separate speculative compatibility |
| NGRAM | Default 12-draft point fails state budget before READY | D4/pin4 capacity passes (33,306 pool), but0/8 prompts equal; accept length1.13–1.58 across real prompts; not promoted |
| DSpark NVFP4 | Actual draft loads; fraction .88 with snapshots leaves negative KV budget | ReplaySSM rejects GDN for DSpark/DFLASH; native embedding-target retry queued |
| Speculative verify switches / Triton | Three original DSpark flags are source-inactive: two EAGLE consumers, one KDA-only; descriptors now reject no-op live trials | Applicable EAGLE controls if MTP fits; Triton verify comparison still pending |
| Draft parameter scan | Pending baseline/draft feasibility | Acceptance vs verify cost across block/step settings |
| Embedding NVFP4 | Actual target converted; patch 0002 required for dispatch. Live capacity/API/retrieval pass; pool 85,299, retrieval budget 84,787 | Full lossy quality; conservative envelope; metadata-only control before attributing full VRAM difference |
| DFlash NVFP4 | Real draft converted to 1.55GB; attention/MLP W4A16 with BF16 activations; fused-loader fixture passes | Draft GPU weights load (1.89GiB stage delta); inference blocked by state sizing; ReplaySSM rejects this GDN combination; native embedding-target retry queued |
| Reference / QAT / more aggressive backbone | Reference and QAT downloaded; reference.94/chunk2048 reaches READY but only5972 pool, failing8192/1024; QAT GPU load pending | Same-protocol full quality/capacity/benchmark evidence; smaller-chunk reference retry queued |
| MTP quantization / token map | Actual reference MTP W4A16 conversion849.4→314.3MB; native Qwen constructor/fused-loader fixture and independent artifact invariants pass. Patch0004 early sharing passes24576-token GPU capacity at.92; strictgreedy0/8. EAGLE map consumer traced; DSpark/DFlash have none | Native/deterministic controls queued; equality, acceptance and speed pending. Reduced-vocab rejection sampling rejected; packed head/map scales require separate compatibility evidence |
| Mixed lookup/model draft patch | No patch justified before profiling | NGRAM value and measured draft bottleneck before implementation |
| Prefix policy / effort / harness | Both lazy and no-buffer+disabled-overlap pass36 matched dialogue/control requests and near-limit retrieval; maximum follow-up TTFT246.8/228.3ms | Separate full effort comparison and corrected actual model/tool harness; pinned four-slot controls do not prove automatic state savings |
| Final combinations | None promoted | Full quality/bench/7200s mixed soak and restart/contention checks |

The context ≥2× and speed ≥1.5× targets remain hypotheses until phase 8 profiling
and controls exist. Existing upstream paths are not claimed as project gains.

## Completed baseline quality and subsequent gates — 2026-11-10

- Official full16k GSM8K:1285/1319 (97.4222896%), two truncations, zero evaluator
  errors; independent1319-row/cap/aggregate validation retained in
  `results/quality/baseline-full-16k-gsm8k-validated.json`. Evaluation throughput
  is not clean serving throughput. Auxiliary sampler logprobs CUDA OOMed after
  44/120 PPL windows; actual harness/ladder skipped. Recovery71037 uses upstream
  input-logprob chunk128, original weights and no new GDN precision flag.
- R3 metadata-only mixed control0/8 vsflat target; embedding DSpark0/8 vsits
  own control; smaller-target DFlash still budget failure. Metadata dispatch
  changes numerics, so total VRAM difference is not pure embedding attribution.
- Legacy GDN isolation localizes BF16 beta casting/state persistence. Opt-in
  patch0003 makes all16 actual decode/verify cases bit-exact; actual NGRAM
  capacity passes but strictgreedy remains0/8. No lossless/speed promotion.
- Lazy-prefix36-request controlled dialogue passes; follow-up median48.2ms,
  max246.8ms, identical flushed median673.9ms. No-buffer requires disabled
  overlap and was rejected before READY; corrected trial remains separate.
- FP32state32689pool and NVFP4KV52096pool pass initial capacity/API/retrieval;
  full quality/speed pending. Reference fails FlashInfer JIT link, not measured
  capacity: missing unversionedcuBLAS aliases now fixed in workspace overlay.
- ExternalMTP weights load2.93GiB beyond target17.46GiB; target-pool budget
  fails at.92 before the existing late embedding/head sharing call. No serving.
- Case11950k-budget follow-up stops8124tokens, reasoning prefix differs from
  preserved16k trace. No20–50k reproduction; case1035 remains for follow-up.

Opt-in patch0004 addresses the measured late-sharing lifecycle: external
Qwen MTP embedding/head sharing now precedes target-pool sizing under explicit
synchronous TP1/PP1 guards. The real converted-Qwen fixture verifies whole
NVFP4 head/scales identity, embedding identity and unchanged draft parameters;
42 regressions pass (`results/tests-mtp-lifecycle-regression.txt`). GPU saving,
capacity, losslessness and speed are unverified;54552 waits for38027 cleanup.

Auxiliary recovery completed:120 PPL windows EN11.16524665/DA11.27067476/
code3.60787577 (all7.89021566),200greedy nonthinking GSM192/200=96%, seven
768-token truncations. Baseline8-prompt strict equality passes. Actual OpenCode exits1
with `provider.no-route` before model requests; artifact unchanged and independent
fixture tests fail as expected. Ladder/retrieval skipped by stop-on-failure.
Evidence: `results/quality/baseline-logprob-chunk128/battery.json` and
`results/harness/baseline-logprob-chunk128/harness.json`.

Corrected no-buffer/disabled-overlap trial passes capacity,8-prompt equality,
all36 dialogue/control requests and near-limit retrieval: follow-up median
47.9ms/max228.3ms, identical flushed median666.5ms. This couples scheduling
and cache policy and pins the same4 slots as lazy; no speed or automatic-slot
saving claim. An added nonsecret environment connection made CLI registration visible, but
the subsequent actual run still failed. Later binary/source and HTTP canaries
identify inheritedPWD as the cause; registration alone did not prove routing.

The repaired unmodified-reference trial reaches READY at.94/pin4/chunk2048,
but its5972-token KV pool fails the required8192/1024 gate. Weight stage
20.20GiB; prefill graph stage.99GiB and reportedfree0.00GiB. No reference
quality/benchmark result is established, and fallback policy remains unverified.
Evidence: `results/probes/reference-link-recovery-094/probe.json`.

## Current execution evidence — 2026-11-10

- C2 baseline passes32107-token capacity, both8192/1024 requests and observed
  two-active telemetry:75 samples spanning15.96s. New gate rejects queued-only,
  missing and single-sample activity.43 regressions pass. Full present grid72214
  is live; no-fallback policy is unverified.
- MTP early-share0004 passes24576-token capacity at.92 where late sharing
  failed. Greedy remains0/8; no speed admission. This is a measured lifecycle
  feasibility repair, not a speculative-losslessness result.
- Case1035's50k-budget replay stops11412 tokens after167.99s;119 stops8124.
  Their answers99076.92/40 disagree with gold95200/35. Prefixes differ from
  original16k traces; no aggregate replacement or20–50k reproduction claimed.
- OpenCode model-list registration was insufficient. Its run implementation
  uses inheritedPWD to chdir, overriding Popen(cwd). Corrected actual local HTTP
  canary reaches /v1/chat/completions for the configured model and receives an
  intentional400. Model/tool fixture still pending; post-grid9452 owns the retry.
- Full-GSM preflight renders all1319 upstream prompts with the active tokenizer.
  Real-data boundary16646 rejects and16647 accepts at16k output; longest263.
- Read-only NVAPI enumerates130 registered names; fallbackID absent, no CUDA/
  sysmem/fallback name. This does not certify private-policy state; no Windows
  setting changed. Evidence: `results/windows-fallback-enumerated.json`.

The unauthenticated actual HTTP canary passes with correctedPWD and no
credential (`results/opencode-route-local-noauth-canary/result.json`), so the
unnecessary placeholder was removed from the final config/harness. Installed
CLI source/hash recorded in `results/opencode-cli-working-directory-contract.json`.
Both long replays are independently scored incorrect by the pinned official
MathEvaluator; their outputs do not replace the official1319-row run.
QAT direct-route1MiB probe timed out; environment route returns the expected
206/prefix hash. Live transfer remains unchanged (`results/qat-route-comparison.json`).

Independent auxiliary audit corrects the earlier zero-truncation report:7/200
responses reach the768-token cap;192/200 accuracy is unchanged. All120 current
input-logprob arrays exactly equal the older120-window PPL-only control and
all44 retained windows before the OOM (max sum difference0). Coverage40 windows
per language, raw PPL sums/counts and greedy booleans/aggregate were recomputed.
`results/quality/baseline-auxiliary-independent-validation.json` retains the
source hash and findings; historical failed-suite status is preserved.

Patch0005 isolates tree Conv1D accumulation order. Legacy CUDA output8/16 exact,
state16/16 exact; corrected output/state16/16 exact; default-off reproduces
legacy output exactly. Full NGRAM0003+0005 still strict0/8; no speed admission.
`results/validation/gdn-conv-correction-summary.json` retains the scoped evidence.
All47 CPU tests pass after default-off, deterministic-control and soak evidence
changes. Soak now rejects missing cases, insufficient duration, missing GPU/host
coverage, gaps above10s and monitor failures. No7200s live soak exists yet.

Post-grid95548 replaces the verified childless waiter4521 before any GPU work.
It adds paired deterministic FlashInfer/Triton non-spec controls, then independent
EAGLE/NGRAM strict-token gates. Only strictly matching, measured faster candidates
are prioritized for full matched quality; this is not recipe promotion.

QAT HTTP87891 completed exit0; all five shards and pinned manifest are complete.
Independent verification passes all five SHA256 hashes against pinned download
metadata, exact2687-tensor index coverage and contiguous in-file offsets.
The index total_size includes345,944 header bytes; file20,559,281,336B versus
payload20,558,935,392B. This exporter convention is recorded, not a missing-tensor
failure (`results/qat-download-independent-validation.json`).
The disposable NVAPI private-ID SetSetting test returned−160 before any save;
fresh-session reads are unchanged (`results/windows-fallback-private-unsaved.json`).
Host GPU execution remains available while the restricted sandbox reports blocked.

Serial waiter9452 now includes baseline vision-enabled six-repeat benchmarks,
7200s mixed short/long/tool/reasoning/unique-image soak, post-soak retrieval and a
fresh model-process restart before candidate comparisons. Earlier waiter95548
was confirmed childless, terminated and archived before any GPU work; the scan
continues unchanged. These stages are queued, not completed stability evidence.
Startup timing explicitly retains OS file/JIT caches and is not called a cold
disk or host reboot. Windows contention/absent-desktop verification is separate.
The queue's completed-GSM comparison helper passes an actual1319-row baseline
control check while preserving that probe's failed auxiliary-suite status;
all47 CPU regressions pass (`results/tests-post-grid-stability-queue.txt`).

The high-fraction no-buffer scan exposes a measured performance cliff: C1,
chunk2048,.94→.96 increases near-limit input47741→54724 tokens (1.146×), but
elapsed7.20→218.60s (30.36×). The fixed8192/1024 request changes19.25→90.22s
(4.687×). Both capacity points pass with the same engine diff. Passive PCIe
samples during the slow point show large transfers; desktop/residency remain
uncontrolled, so this is consistent with paging, not proof of fallback or formal
throughput. The scan is preserved, with observer16474 collecting later points.
Evidence: `results/envelope-slowdown-observation.json` and
`results/envelope-pcie-observer.jsonl`. No high-fraction balanced recipe promoted.
Completed-request timestamp windows exclude startup transfers in
`results/envelope-pcie-phase-observation.json`: .96/.98 C1 fixed-shape requests
take90.22/85.96s with sampled mean RX10,779.5/9,275.1MB/s and
TX12,080.5/9,708.9MB/s. Subsequent.88 C2 completes both requests in20.08s with
two-active proof and RX/TX86.0/31.5MB/s. SM clocks are near2932–2947MHz across
these samples. Different fraction/concurrency/desktop states and2/10/2 counter
rows do not support a causal fallback conclusion or a formal throughput claim.

`bench/compare_battery.py` validates raw120-window PPL,200 greedy response-text
answers/scores and truncation counts before reporting matched auxiliary deltas.
It rejects changed corpus/tokenization, partial windows, false stored scores,
hidden truncations and nonfinite golds. The actual baseline control passes with
192/200,7 caps and zero deltas; no auxiliary tolerance or final promotion is
invented. Four new tests bring the CPU suite to51 passing tests; focused verifier
rerun passes after the final nonfinite-gold guard.

Actual checkpoint ordinary-FP4 row diagnostic87167 completes160/160 exact output,
packed-activation and logical-scale comparisons across eight representative
linears, M1/2/4/8/32, two synthetic activation magnitudes and ATen-invariant modes.
Uses actual ModelOpt weight creation/preparation/apply;3.86s diagnostic wall time,
996,719,616B peak allocated. Runtime fused GDN QKVZ/SwiGLU/proj-conv and complete
model activations are outside this scope. No FP4 padding patch is justified.
`results/validation/fp4-row-parity-r1.json` and clean-boundary guard preserve
source hashes and same-PID scan resumption. BF16 gates are a separate operator;
diagnostic72548 completes112 finite cases using actual `[b;a]` weights from
layers0/1/30/62, two activation magnitudes and M1/2/4/8/32/128/1024. Native
UnquantizedLinearMethod is exact14/56 (M4:2/8); upstream ATen invariant mode is
exact56/56. Maximum synthetic native discrepancy0.0625, atM4 max0.0009765625.
Coverage/finite/exact counts are independently recomputed in
`results/validation/projection-parity-summary.json`. This supports the queued
matched deterministic controls, without proving full-model causality, equality,
quality or speed. The same scan resumes; no new BF16 kernel patch is needed.

First reusable measured-envelope table:95 raw points,15/25 completed searches
in `results/envelope-snapshots/20261110-1000/table.md` and its JSON. The summarizer
validates CSV/probe outcome consistency, actual near-limit/fixed token counts,
finite positive times and independently reparsed C2 raw metrics. Pending groups
remain pending; high-fraction latency variation is visible beside capacity.
No maximum/quality/speed/recipe promotion is inferred from this partial table.

Full baseline official regrade24714 completes all1319 saved generation texts.
Question/gold pairs match the actual cached upstream dataset, and the scorer
receives no stored predicted_answer/symbolic_correct fields. All1319 extracted
answers and correctness labels agree;1285 correct (97.4222896%). This validates
the existing generation and does not replace it. Scorer/verification/input hashes
and all regraded rows are in
`results/quality/baseline-full-16k-official-rescore.json`.
All nine tokenizer/template files still match the existing all-GSM8K parity
proof across target/reference/QAT and five thinking modes; file identity, rather
than new rendering or generation, is rechecked in
`results/tokenizer-parity-file-recheck.json`.

After all18 cached BF16 combinations completed, the first radix-disabled
.88/chunk2048/C1 point passes62,109-token capacity with one logical Mamba slot.
Its59,002/1 request takes9.04s and8192/1024 takes16.37s; sampled device peak
22,697MiB, before-load usage1197MiB. This is an individual capacity observation,
not a maximum or final max-context recipe; strict greedy, natural retrieval,
full quality and stability remain separate gates. Raw point:
`results/probes/2026-11-10T072544.182031_0000/present-disabled-2048-c1-bfloat16-0.88000/probe.json`.

Bounded NVAPI follow-up70370 tests private SetSetting in both profile handles:
both return−160. The standard power DWORD read also returns−160, so its two
same-value control writes are skipped. No SaveSettings call or persistent change;
fresh fallback reads are unchanged. This test cannot establish that the public
setter works or prove private-only rejection. Policy remains unverified;
`results/windows-fallback-setter-controls.json` preserves all four outcomes.

Endpoint-first resume85433 intentionally replaces old72214 at a clean boundary,
after the active.98 probe completed and all model servers exited. Because its
parent was stopped, that completed child had not yet been appended; captured
PID/output identity and raw results are committed during handoff. All114 points
and every prior CSV byte are preserved, with sources/metadata/logs archived and
all114 raw probes independently revalidated.19/25 combinations are complete;
the resumed partial search uses six retained points and zero new GPU calls.
Same25 combinations,.88-.98 bounds and full per-point request/activity gates.
Low/high endpoints precede intermediate probes; high failure retains coarse
discovery and bisection.55 CPU regressions pass. Source/protocol handoff, rather
than timeout recovery: `results/envelope-endpoint-handoff.json`.
The existing post-grid9452 process remains waiting; observer16474 completes
against the old scan owner and does not silently follow the new PID1409798.

Delivery patch stack independently verified in a fresh temporary tree at the
exactc3c292b65 base: all five patches apply sequentially and reproduce all nine
tracked engine changes byte-for-byte. No uncovered tracked changes or untracked
source files; live checkout unchanged. Patch/file hashes retained in
`results/patch-stack-independent-validation.json`; recipe setup now applies the
complete stack. This is source reconstruction, not new inference/quality proof.

Serial waiter24867 replaces verified childless9452 before any GPU work. It adds
one embedding-target deterministic FlashInfer control and native DSpark/DFlash
strict gates motivated by the actual BF16 row-batching discrepancy. Actual probe
dry-runs confirm matching model/backend/chunk/fraction, eight Mamba slots and
24576-token budget after profile overrides. Both candidates enable0003/0005;
quantized-embedding MTP early sharing stays off. The fixed argument hooks have
no direct deterministic-mode prohibition for these CUDA paths; live capacity,
API/retrieval and strict equality are still required before timing/adoption.
Vision stability now uses the measured chunk1024 configuration. Full-GSM stage
deadlines increase8h→24h while keeping1319 examples and16k output unchanged.
All56 CPU regressions pass (`results/tests-deterministic-embedding-queue.txt`).
Queued work is not success; no model/quality/speed/stability promotion follows.

Present-desktop grid85433 completes exit0:25/25 searches,126 passing raw probes,
all upper endpoints.98 passing. Independent final summary validates every actual
request/token count and reparses all C2 activity metrics; the retained114-point
prefix is unchanged. `results/envelope-present-validated/table.md` and its JSON
are the completed table. No global/physical-VRAM maximum, no-desktop condition,
fallback enforcement or final recipe is certified. Post-grid24867 starts its
first baseline follow-up only after scan cleanup; capacity, original8-prompt
equality and natural retrieval pass, with the real ladder underway.


Latest post-grid correction: the actual OpenCode model completes read/edit/shell
and independent unchanged fixture tests pass. All48 real-prompt ladder rows
complete; client concurrency2 uses a max-running1 server and includes queueing.
Vision chunk1024 capacity passes twice, text-control equality0/8 twice, own-vision
restart equality8/8. Original strict failures prevented any soak. Corrected61678
uses the own-vision control for7200s stability without transferring text quality.
FlashInfer deterministic control originally fails the4096 alignment constraint;
all corrected FI profiles explicitly use tile1024. Triton control passes while
its EAGLE candidate fails strict own-control equality. The completed child is
retained at a verified clean handoff, with no in-flight test interruption.
57 runtime-environment regressions pass. Raw handoff:
`results/post-grid-corrective-handoff.json`; live source:
`results/driver-sources/post-grid-validation-resume.py`. No candidate promotion.


Windows application contention: added a bounded owned D3D11 random-texture/copy
fixture, confirmed native C# compilation and SDK structure sizes without GPU API
calls, and passed59 CPU regressions. Three before/during/after8192/1024 observations
and API/strict/retrieval recovery are queued under tool16381 after verified owner
61678 exits. No live contention result or physical-residency claim exists yet.
Source: `bench/windows_contention.py`, `scripts/windows-gpu-occupancy.ps1`.


Vision r2 formal benchmark: all six required repetitions validate, throughput
66.244597/62.667567tok/s at1k/8k and mean per-repeat median TPOT15.017846/
15.300495ms. This mode is not a paired text-baseline speedup.7200s mixed soak is
live; first frozen curves/raw sources independently validate409 samples and166
responses with20 distinct images, zero raw-response errors, but no full soak gate.
Five eight-prompt speculative/control comparisons independently validate all40
common-prefix lengths. First emitted token/logprob is exact in all40; chosen-token
logprobs diverge at position1 in all40, while full equality stays0/8 per group.
No causal operator or new source correction is established by these observations.


Vision mixed-soak r2 negative result:795.3556s,311 completed requests,38 successful
vision frames; the39th vision attempt fails the expected `red square` literal
check.763 telemetry rows are frozen; full7200s gate is false. Original failed
caption was not retained, so cause remains unknown. Future failures now retain
raw caption/digest without relaxing checks. All38 previous frames regenerate
byte-exactly; inferred frame38 is being tested in a fresh owned process under
2469, with the same61678 controller resumed afterward. Restart r2 and corrected
FlashInfer deterministic control pass; its EAGLE candidate fails strict equality.


Fresh Frame38 reproduction: own-control8/8 passes, but the old caption request
returns two `<box>` coordinates. Same-image prompt ablation with cache flush
returns `red, square, blue, circle` under a neutral words-only request. Literal
substring matching incorrectly rejects punctuation between adjacent correct
words; semantic recheck accepts the actual color/shape associations while
rejecting red-circle/blue-square. Both original negative probes remain intact.
63 CPU checks pass. Live swapped-color controls and a conditional full7200s
mixed rerun are owned by95962; same61678 controller resumes after its clean
window. The preceding SIGSTOP-acknowledgement guard exited before GPU work;
v3 corrects the asynchronous acknowledgement wait, without interrupting a test.


Caption correction live gate now passes: old same-image request returns boxes,
neutral words-only request returns correct adjacent color/shape words; the
blue-square/red-circle counterfactual also passes, with all five text parser
checks. Full r3 startup validation is active before its7200s mixed soak.63 CPU
regressions pass. No original failure is reclassified as a completed soak.


r3 vision negative:565.7376s,215 successful requests,26 verified images; next
caption is `red, square`, missing blue circle. Exact image hash reconstructs and
fresh-process Frame26 repeats the omission; five text API checks and own-control
8/8 pass. Full quality gate stays false. Runtime observation84867 now completes
the requested duration while retaining semantic errors; no HTTP/OOM/monitor error
is ignored. Its early frozen1547s snapshot contains1485 samples,616 successful
requests andone confirmed failed image,77 unique hashes. Both duration gates are
pending.66 CPU regressions pass. This cannot convert bad captions into quality
passes or replace final recipe validation.

Fused activation/FP4 diagnostic33501 completes112/112 exact batched/sequential
packed data, scales and down-projection outputs; separate-path outputs also112/112.
Coverage/metrics independently validated. No full-model parity or speed claim.
Paired eager control/EAGLE/NGRAM guard76613 is queued after runtime84867, requiring
strict own-control equality before speed and matched full quality before later
adoption. No new source kernel patch or final promotion.

## 2026-11-11 session recovery

Host GPU access works; restricted NVML fails because the sandbox lacks dxg. All prior model/controllers are absent after external session interruption. Exact prior manifests are archived in `results/recovery-history/20261111-session-interruption/`. Observation r1 is only3931.935s and retains one vision error; no7200s completion claim or duration stitching. The completed deterministic embedding DFlash r2 failed token validation after passing capacity; its raw terminal evidence was adopted with null exit code (original wait status unavailable).

The project Python selection fix is verified and retained: project sgl_eval0.1.3 imports, owned validation commands rebind, and no original environment is changed. New serial controller source is `results/driver-sources/serial-recovery-20261111.py` (tool4056). It owns continuous runtime, Windows contention, eager gates and retained quality queue in order. Recovery tests forbid budget retry after a capacity-passing validation failure. All73 CPU checks pass. New prelaunch GPU usage3299MiB differs from old1245MiB; vision.88 startup fails with hybrid state cache too small (3slots,0requests). The.92/pin4 trial is new evidence, not a paired reuse of the old cohort. Windows still drives4K on NVIDIA; no absent-desktop/fallback confirmation.

### Bounded actual-forward tracing prepared

New `bench/forward_trace.py`, `capture_forward_trace.py`, `compare_forward_trace.py`
and `trace_pair.py` implement a256MiB, two-forward numerical diagnostic using
public forward-hooks. Source audit shows the eager runner calls outer `.forward`
directly, so the actual hook target is backbone `model`, not the root or weight
prefix. Capacity requests stay unarmed; the short real prompt captures last
prefill/first verify rows and GDN persistent after-prefill/before-step states.
No GPU diagnostic has run while the soak owns the model. Four actual PyTorch/
SGLang-hook CPU regressions and the full77 checks pass; actual probe CLI/factory
resolution proof is in `results/forward-trace-cli-validation-20261111.json`.
This creates localization capability, not a parity, causal or speed result.

### A-layer completion audit and queued numerical comparisons

Current-state audit confirms the original cuDNN FP4 and FlashInfer GDN decode/
prefill trials passed startup/capacity and captured8 prompts, but stopped before
benchmarking on strict token differences. They have no speed/full-quality
result. These are non-speculative numerical backend changes; a separate,
explicit numerical evaluation is required rather than treating the strict
negative as a measured performance loss. Original configs/negative evidence
remain unchanged; no exception is applied to speculative/caching paths.

`results/driver-sources/numerical-backend-followups-20261111.py` is queued as
tool40822/PID233424/start2262817. It waits for verified live4056/PID106665/start
2114624, requires its completed_queue status and an empty GPU/port boundary,
then brackets the three candidates with strict original-baseline controls.
Six raw benchmark files are independently reparsed for each trial. Full quality
is prioritized only when both mean gains are at least2percent and candidate
replicates exceed the complete observed bracketing control range. This is not
a significance or final adoption claim. Full FP32-state1319/16k quality is also
queued; the prior queue covered state precision capacity/speed only.

All78 CPU checks pass, including rejection of one-load wins, overlapping noise
and incomplete controls (`results/tests-numerical-followups-20261111.txt`).
Read-only host PID/start/cwd/command proof is retained in
`results/numerical-followup-owner-check-20261111.json`; the sandbox process
namespace cannot see the host owner, so a sandbox check is not authoritative.
No model process was interrupted or overlapped by this followup.

### Resumed-session hardware observation

Three passive whole-device windows during the unchanged vision runtime r2,
without any server request or configuration change, show P0/SM2880–2932MHz/
MEM14001MHz/49–51°C and561–568MiB free. Six PCIe samples span RX6119–23454
and TX7270–20589MB/s. This observes high clocks and large traffic, not
process-specific paging, residency or a verified Windows fallback setting.
The slower47.54s8192/1024 capacity request is a different desktop cohort and
not a formal repeated speed benchmark. Raw command output and verified owner
are in `results/vision-runtime-r2-hardware-observation.json`; parsed ranges
and raw SHA256 are in `results/vision-runtime-r2-hardware-summary.json`.
The continuous run is not restarted or modified.

### Frame26 reproduced in the new.92 runtime cohort

At about29min, the new process again returns only `red, square` for image
SHA256cb238edee2c25d820619700d40daf8e329edcb9fcf71d43d5800ee5bd9ccc5c9,
with normal stop. Four retained cases (r3, fresh repro, interrupted runtime r1,
new runtime r2) have the same image hash, recorded neutral prompt and omission.
An independent adjacent-word check confirms red square present and blue circle
absent in each. The fresh process proves long runtime is not required; changing
from.88 to.92 does not eliminate this fixture error. This is not a population
error rate or causal-operator conclusion. Raw/probe pointers and file hashes are
in `results/validation/vision-frame26-cross-process-reproduction.json`.

The frozen new snapshot at1760.31s has1691 telemetry samples and one retained
vision failure, with no observed startup-through-snapshot OOM log events and
no request/runtime exception. Its full mixed quality gate remains negative; the
unchanged continuous7200s runtime observation continues. Snapshot artifacts are
in `results/soak-snapshots/vision-runtime-observation-r2-092-pin4-frame26-error/`.
The actual.92 probe's prelaunch GPU use is3314MiB;3299MiB was the earlier host
snapshot before the resumed probe sequence, not this second probe's exact value.

### Live probe handoff after tool session transition

Tool4056 became unavailable. Host process checks found the serial parent
PID106665 and numerical waiter PID233424 absent, with probe111678/start2118404,
server111705/start2118417 and soak141424/start2153004 still running.
`results/recovery-history/20261111-live-probe-handoff/handoff.json` preserves
their identities and the archived controller manifests and source hashes.

The new controller PID389274/start2487970 has its own session and no terminal.
It waits for the original probe's terminal evidence and an empty GPU boundary,
then resumes the queued restart, Windows, eager and post-grid trials. Numerical
followups will run in the same sequence under this controller. The surviving
soak continues from its original start. The prior controller's child exit code
is unavailable, so adoption will record the raw terminal status and a null exit
code. All78 CPU tests pass after the handoff changes.

### Two-hour mixed runtime completed

The original probe survived the handoff and completed its uninterrupted
7,226.53s run. Independent validation confirms 913 successful responses, all
8 case types attempted, 114 unique image requests, one retained Frame26 caption
omission, 6,826 valid GPU/WSL samples with maximum gap 3.183s, and zero observed
OOM events in the frozen server log. Frozen soak/telemetry/log SHA256 values
match the report. GPU free memory ranged from 0.499 to 0.782GiB and WSL available
memory from 21.478 to 22.326GiB. The report and curves are under
`results/soak-snapshots/vision-runtime-observation-r2-092-pin4-final/`.

Post-run 24,064-token retrieval passes all three depths and records prefix
hits. Warm TTFT is 1.853/1.128/1.015s at the three depths, with one sample
above 1.5s. The detached controller recorded the original probe as passed
from its finished raw result and retained a null process exit code. It then
started the planned fresh restart validation.

### Fresh restart and Windows contention passed

The vision .92/pin4 restart reaches READY in 148.75s and serves a 24,576-token
pool. It matches8/8 control captures, passes API smoke and retrieves at all
three depths with warm TTFT0.148–0.606s. The original boot took51.10s. These
are fresh-process times with existing OS/JIT caches.

The text .88 Windows contention probe reaches READY in77.07s with28,139 tokens.
Native D3D11 PID20300 allocates536,870,912 nominal bytes on the 5090 D v2,
completes50 copy passes, and reports normal release and exit0. All nine
8192/1024 requests complete. Median elapsed times are16.912/16.910/16.772s
before/during/after. Device-used samples span23309–23348/23446–23480/
23310–23346MiB. Independent checks compare all8 strict captures before and
after to the original control, verify request timestamps against the native
lifetime, parse all143 phase telemetry rows, and verify API and27,627-token
retrieval recovery. The owned log has zero observed OOM events.
`results/contention/baseline-windows-contention-r1/independent-validation.json`
retains raw hashes and phase evidence. GPU residency and fallback policy remain
unverified. A fresh CIM query still shows NVIDIA driving3840×2160.

### Eager gates and trace scheduling

The eager control passes six formal repeats at13.875/13.615tok/s. Eager EAGLE
and NGRAM each pass24,576-token capacity and complete8 greedy captures, with
0/8 matching their own control. Native EAGLE, packed EAGLE and packedD2 also
record0/8 exact outputs. Independent raw extraction preserves real-request
acceptance and throughput separately from the synthetic capacity request in
`results/speculative-eager-native-observations-20261111.json`. Native EAGLE
real requests report weighted accepted length2.821 and median decode124.993tok/s.

Automatic approval review rejected a proposed SIGSTOP of the post-grid controller
to borrow a trace window, citing possible disruption to the running queue. That
script was removed before execution. The safer detached waiter
`results/driver-sources/trace-after-queue-20261111.py` waits for the original
queue to exit naturally, then invokes the existing trace entry point. It is
live asPID558575 and records `results/eager-ngram-trace-after-queue.json`.

### FP32-state and NVFP4-KV formal benchmarks

Both experiments complete six formal repeats with10 exact1024-output requests
each. Independent reparsing of all12 files reproduces the stored summaries.
FP32-state reports36,506 pool tokens, prelaunch2250MiB, peak23598MiB, and
66.091/62.635tok/s at1k/8k. Median TPOT across the three repeats is15.047/
15.346ms. NVFP4 KV reports57,280 tokens, prelaunch2254MiB, peak23996MiB,
66.328/63.594tok/s, and15.017/15.072ms median TPOT. Their full quality gates
remain pending. Raw files and hashes are indexed by
`results/bench/precision-contrast-validation-20261111.json`.

The post-grid queue advanced to reference-small-chunk-full. Native EAGLE D3
and DFlash .92/packed failed strict tokens. DSpark/Triton with the embedding
candidate reported8297 tokens, below the required8192+1024 capacity request.
All negative evidence is retained in `results/post-grid-validation.json`.

### Complete reference GSM8K and target comparison

Reference-small-chunk-full completes1,319/1,319 GSM8K rows with1,278 correct
(96.8916%),4 truncated at16,384 tokens and0 request errors. The initial target
scored1,285 (97.4223%),2 truncated and0 errors. Independent matched comparison
verifies generation parameters and all question/gold pairs. Taking RadixArk
as the reference control, the all-linear NVFP4 target gains16 correct answers
and loses9, net7 or0.5307 percentage points in these single runs.
`results/quality/target-vs-reference-matched-gsm8k.json` records the correct
comparison direction, row disagreements and raw hashes. Reference auxiliary
quality is running with all120 PPL windows complete.

### Reference auxiliary evaluation complete

The reference passes all120 PPL windows and200 greedy requests, scoring190/200
with7 cap hits. The target baseline has192/200 and7 cap hits. Independent
matched raw comparison verifies identical tokenized PPL windows, questions,
golds and protocols. Reference/target overall PPL is7.7335518/7.8902157,
a target increase of2.0258percent. Language deltas are+1.6996percent EN,
+2.2720percent DA and+2.0460percent code.
`results/quality/target-vs-reference-auxiliary.json` retains all recomputed
metrics and source hashes. The queue advances to mixed-control-full.

### Mixed-format control full quality

Mixed-control-full finishes1,319 rows with1,278 correct,0 truncations and
0 request errors. Relative to the initial target's1,285 correct, the measured
drop is0.5307 percentage points and exceeds the0.5-point gate. The complete
auxiliary comparison verifies120 windows and200 greedy examples, scoring
192/200. All120 raw PPL logprob windows match the initial target exactly.
Six greedy response texts and191 answer extractions match across200 examples.
The numerical/quality difference is retained with its control protocol.
`results/quality/mixed-control-full-gsm8k-comparison.json` and
`results/quality/mixed-control-full-auxiliary-comparison.json` retain the raw
hashes and disagreements. Embedding-full is now running.

## 2026-11-12 host restart and quality resume

After the explicit pause/resume, the host boot ID changed to
c695701d-5f39-4788-8e78-f7f6a5f9fee4. No old controller/probe/model process
remains. Embedding-full retained468 complete unique examples. Source files
and hashes are archived in `results/recovery-history/20261112-host-restart/`.

`bench/resume_quality.py` reuses the upstream runner, prediction writer and
official MathEvaluator. It reconstructs retained Samples from their original
responses, budgets and timestamps, regrades all examples, and generates only
the851 missing IDs. It verifies the original recorded protocol, checkpoint
revision, engine diff, dependency versions, prompt/template and question hashes.
Final quality retains1319 examples and one repeat at16k. Resumed timing metrics
are left null. All81 regressions pass, including duplicate/mismatched input,
cap, seed, official rescoring and one-transport-call-for-one-missing-ID tests.

The detached `results/driver-sources/resume-validation-20261112.py` runs as
PID54183/start963572. Its first owned probe is
`embedding-full-resumed-20261112`; once quality completes, the logical
embedding-full queue entry will point to this measured result. Existing
partial raw data is preserved. The remaining post-grid, numerical and trace
work follows through the same detached controller.

The resumed live run passed all source/checkpoint/protocol checks and officially
regraded the468 retained examples. Independent comparison confirms complete
record equality, including generation text, timestamps, token counts, extracted
answers and scores. Fresh model generation has started for the remaining IDs.
`results/quality/embedding-resume-retained-validation-20261112.json` records
the original snapshot hash and retained-row checks.

### Forward trace queued at the normal post-grid entry

The running resume controller starts the post-grid child only after the owned
embedding probe finishes and its logical completion is recorded. That normal
entry now invokes the eager trace before the remaining full quality trials.
The condition checks the parent PID, running_post_grid_child state and completed
embedding evidence. The trace entry checks model processes and port ownership.
Two regression cases verify that active quality, another parent or missing
completion keep the trace queued, and that normal child entry launches it.
All83 checks pass.

The original post-grid driver is archived at
`results/driver-sources/archive/post-grid-validation-resume-5477d728.py`;
`results/normal-queue-trace-handoff-20261112.json` records both source hashes.
The late trace invocation can reuse a completed result after verifying collector
and engine source plus raw tensor comparison. The active evaluation continues
with its loaded source.

### Delivery documentation reconciliation — 2026-11-12

Rechecked the terminal raw harness, restart, contention and frozen two-hour
report before removing stale pending statements from docs/recipe.md. The
7,226.53-second runtime gate passes while the retained caption-quality gate
fails. Actual OpenCode tools, fresh-process recovery and owned Windows
contention are complete for their recorded configurations; selected final
combinations, fallback certification and absent-desktop measurements remain
open. Local recipe evidence links resolve. docs/status.md now distinguishes
completed reference/mixed-control quality and failed eager/native strict gates
from the running resumed embedding evaluation and queued actual-forward trace.

### Post-restart display state — 2026-11-12

Read-only Win32_VideoController refresh reports NVIDIA2560×1600,
GameViewer3840×2160, and no active AMD resolution. The NVIDIA display therefore
remains active in the host report. No display/policy setting was changed; this
is not absent-desktop or physical-residency evidence. The raw controller fields
are retained in results/windows-display-20261112.json. Full embedding quality
continues in the original detached serial controller.

### Full-scope working checklist — 2026-11-12

Added docs/completion-checklist.md mapping all deployment/optimization stages,
A/B/C/D items, protocol gates, named artifacts and conditional/optional work to
recorded evidence and explicit gaps. All local artifact links resolve. This is
a working checklist, not a completion claim: failed speculative equality, the
missing physical cohort, pending full quality and final-combination reruns remain
open. Current full embedding generation continues without interference.

### Quantized trace layouts remain opaque — 2026-11-12

Actual ModelOptFp4LinearMethod.apply consumes packed uint8 activations and
interleaved floating-point scales. The pending collector previously treated
all floating-point leaves as dense, which could falsely localize a difference
from unaligned scale slices. The collector now propagates opaque scope through
quantized containers, limits dense comparisons to ordinary FP16/BF16/FP32/FP64,
and labels physical opaque excerpts explicitly. The reader rejects falsely
labeled quantized dtypes and nonfinite dense values, while retaining opaque
nonfinite padding without a parity claim. Real PyTorch hooks/serialized tensors
cover both FP8 and FP32 scale pairs. All84 CPU checks pass with the project
runtime/cache environment loaded. The initial unconfigured environment failure
is preserved. Serving engine hash is unchanged; no actual GPU trace is claimed.
Evidence: results/forward-trace-layout-validation-20261112.json.

### Independent official candidate rescoring — 2026-11-12

Generalized the frozen baseline verification into bench/rescore_quality.py,
with explicit completed-run/output arguments and no existing-output overwrite.
It checks all actual upstream question/gold pairs, re-extracts from generation
text through pinned official MathEvaluator with predicted-answer reuse disabled,
and rejects source artifact changes during grading. It does not generate new
answers or replace recorded aggregates. Real full reference and mixed-control
runs each regrade all1319 with1278 correct; both extraction and score mismatch
lists are empty. Thus the mixed control remains a0.5307pp gate failure.
Evidence: results/quality/reference-small-chunk-full-official-rescore.json and
results/quality/mixed-control-full-official-rescore.json. Both record the current
verification source hash. Active embedding generation is unchanged.

### Embedding full GSM8K gate passed — 2026-11-12

The unchanged1319-example/one-repeat/thinking/seed1701/16k protocol completed
with1281 correct (97.11903%),3 truncations and0 request errors. Versus the
initial target1285,11 answers improve and15 regress: net−4/−0.30326pp, passing
the0.5pp gate. The metadata-only mixed control scored1278; the embedding pair
is net+3, without a significance or causal-improvement claim.

All468 retained response dictionaries remain exact,851 fresh responses complete
coverage, and resumed aggregate timing fields are null. Independent official
MathEvaluator rescoring re-extracts every saved answer, checks every upstream
question/gold and reproduces all1319 labels/extractions. The full quality run
uses a24576-token pool cap; earlier85k capacity does not become a fully qualified
recipe by this result. Auxiliary120PPL/200greedy evaluation is still running.

Evidence: results/quality/embedding-full-resumed-20261112-independent-gsm8k.json,
results/quality/embedding-full-resumed-20261112-official-rescore.json and
results/quality/embedding-vs-mixed-control-matched-gsm8k.json. The parent's
post-grid state is untouched pending its normal completion/adoption boundary.

### Actual trace request identity — 2026-11-12

The pending capture client now calls the fixed source's /tokenize endpoint
for its exact raw real prompt with add_special_tokens=True before arming.
It retains token IDs/count and requires complete equality with the actual
prefill IDs, in addition to positions and the first emitted root. Native
text generation is unchanged. A CPU negative case supplies a complete-looking
trace with the same positions/eight outputs/root but one wrong prompt ID; it
is rejected while matching inputs pass, with owned-arm cleanup in both. All85
regressions pass. This is request attribution evidence, not a GPU trace or
performance result. Engine source hash is unchanged; active quality continues.
Proof: results/forward-trace-request-validation-20261112.json.

### Embedding full auxiliary completed and normal trace handoff — 2026-11-12

The owned probe completed all120 PPL windows (40 each EN/DA/code) and200
nonthinking greedy questions:195/200 correct,6 caps at768 tokens. The initial
target is192/200 with7 caps. Overall PPL7.86931963 versus7.89021566 is−0.264835%;
per-language changes EN−0.345907%, DA−0.286556%, code−0.157814%. Independent
comparisons check original raw logprobs, sums, token IDs, questions/golds,
response-derived predictions and aggregates. No auxiliary threshold is invented.

The first comparison command incorrectly used system Python3.10.12; three old
baseline window sums differed by5.684e−14 due to summation arithmetic. Repeating
with the recorded project Python3.12.12 reproduces every stored sum exactly for
both runs; original data and checker were unchanged. Reports record the runtime.
Evidence: results/quality/embedding-full-resumed-20261112-auxiliary-comparison.json
and results/quality/embedding-vs-mixed-control-auxiliary.json.

The probe is terminal passed. The serial parent adopted its logical embedding
entry while preserving the original interrupted files, then started the actual
eager control/NGRAM trace through its normal post-grid child entry. No process
was paused or interrupted and no model servers overlapped. Final recipe pool
limits, formal speed, prefix/harness and stability qualification remain open.

### First actual full-attention divergence and verified reuse — 2026-11-12

The owned eager Triton control/NGRAM GPU diagnostic completed, with48 identical
actual prompt IDs, matched positions/root input and8 actual outputs. Each records
prefill plus first decode/verify; control has1 query, NGRAM4. Both graph phases
are disabled by the canonical cuda-graph-config. Captures228,578,320/229,757,968
bytes remain below256MiB. All1780 shared dense prefill boundaries match; all96
shared persistent-before-step GDN tensors match. In the first step,93 shared
dense boundaries precede the first difference at model.layers.3.attn output:
2962/6144 values, maxabs0.0625, with Q/K/V inputs bit-identical. The eight short
output IDs match, which does not replace the retained eight-prompt/512-token
strict failures. Missing inner GDN structure and161 opaque layouts/frame remain
explicit. Whole attention KV/masks were not captured, so no causal operator or
full-model losslessness claim is made.

Fixed-source audit selects one-stage cached unified verify versus two-stage
decode. Both NVIDIA paths quantize Q to the FP8 KV dtype; a BF16-versus-FP8-Q
explanation would be wrong. Isolated replay with identical cache/mask data is
the next source-grounded experiment. The engine is unchanged and NVFP4-KV full
quality continues through the normal serial queue.

Independent raw recomputation exactly matches canonical JSON. A real reuse bug
compared Python tuple signatures to deserialized JSON lists and would falsely
stop the late queue. The corrected read-only runner normalizes them, preserving
the exact measured runner in results/driver-sources/archive/trace-pair-4c00802ee379114ccaaa070fa3cb5a77a6844e14bdfb7f52f4047b2b937ccf57.py.
Collector/client/comparator and engine hashes remain required; changed runners
need the exact archive. Original measurement files were not rewritten. All87
CPU regressions and real retained-data CLI reuse pass without launching a model.
Evidence: results/forward-traces/eager-ngram-r1/independent-validation-20261112.json,
comparison.json and reuse-verification-20261112.json; see
docs/attention-verify-findings.md for source paths and limitations.

### Prepared first-attention cached-KV cases — 2026-11-12

Added attention_case.py as a second opt-in public hook, preserving the measured
forward collector, capture client and comparator. Complete selected-layer Q/K/V,
output, actual logical post-write KV bytes, original storage geometry, mask rails
and decode split metadata are bounded to32MiB. Unsupported MLA/DCP/window/page
layouts fail. CPU hooks exercise noncontiguous physical slots, copy independence,
mask discrepancies, corruption and budget failures. Default trace plans and real
old-data reuse remain compatible. All92 CPU checks pass.

The future numerical child now captures these cases at its normal post-grid
completion boundary, with existing parent/status verification and model/port
checks before and after. Original future source14a5538b… is archived; the active
post-grid helper remains32f6146b… and engine1490ee5b… unchanged. No live task is
paused, replaced or competed with. Concrete command/source hashes are recorded
in results/attention-case-queue-handoff-20261112.json. Extra GPU cases and actual
causal kernel replay are not measured yet.

The active NVFP4-KV probe passed three-depth75,004-token retrieval with observed
prefix hits, but warm TTFT27.200/0.538/18.637s misses1.5s in two depths. Pool
capacity75,520 is an observed cohort, not a certified physical maximum. Full
GSM8K is running; final prefix/performance qualification remains negative/open.


### 2026-11-12 — isolated cached-attention replay prepared, GPU still queued

`bench/replay_attention_case.py` reconstructs original physical FP8 cache slots
and Q/output layouts within a256MiB storage bound, with a separate256MiB
scratch bound. It first requires both original kernels to reproduce every
recorded prefill/step output exactly, then compares the verify queries through
decode using their actual per-query visibility rails. Batched versus separate
query results and the actual control root are independent checks. Failed
original reproduction withholds route attribution. A CPU preflight is never
reported as GPU parity. No serving source is changed.

The future numerical child is updated only while its adopted manifest is
absent and parent54183/start963572 remains in running_post_grid_child. Its
previous source is archived at
`results/driver-sources/archive/numerical-backend-followups-3e692738a7b4fe4cdc22d3344d3a00bda9d12d899f65679ed0bc724e12047878.py`.
It invokes replay only after successful actual cached-case capture; idle
checks bracket it. Active post-grid/controller sources remain unchanged and
no running process is paused or interrupted.

All96 CPU regressions pass, including actual hook capture, noncontiguous Q,
original physical FP8 row restoration, bounded allocation failures,
contradictory duplicate rejection, branched visibility/position validation,
output layouts and independent sequential outputs. The historical measured
trace still passes read-only reuse. Evidence:
`results/tests-attention-replay-20261112.txt`,
`results/forward-traces/eager-ngram-r1/reuse-verification-attention-replay-20261112.json`,
`results/attention-case-queue-handoff-20261112.json`.
Actual cached-case GPU collection and isolated kernel replay are not executed
yet; no causal fix, speed claim or final recipe is promoted.


### 2026-11-12 — passive observation of the continuing KV quality slowdown

The real evaluator887824/start1834936 and model scheduler773813/start1721857
remain live, alongside the original serial owners. Sixty-one complete GSM8K
responses are saved; the next long response is generating. Server log counts
continue to advance, with recent decode throughput6–7tok/s. A short passive
whole-device window observes P1, SM2940MHz,41.7°C, about152W,99–100% GPU use,
and PCIe RX4179–15598MB/s / TX1865–14838MB/s. Neither high utilization nor
PCIe traffic attributes the slowdown to a process, proves fallback/paging or
certifies physical residency. No benchmark/API request, driver change,
process signal, queue edit or evaluation-protocol change is made in this
observation. Raw evidence:
`results/progress-checks/kv-nvfp4-slow-window-20261112.json`.


### 2026-11-12 — real QAT loading and CPU post-load preparation preflight

All496 quantized text projection geometries and1984 packed/scale tensor
headers match the fixed QAT architecture. Its96 GDN b/a projections are
quantized, unlike the initial target's BF16 b/a weights. All992 global
scalars are finite/positive; all176 fused groups have equal component
globals, satisfying the native max-reduction precondition without silently
changing per-component weight scales.

Four bounded real checkpoint projection groups (GDN qkvz, GDN ba, gated
attention qkv, MLP gate/up) pass actual Qwen name-remapping/stacked loaders
and compressed-tensors W4A4 weight creation. Destinations are poisoned first;
all16 loaded parameters match the independently concatenated expected
checkpoint bytes. Native CUTLASS-path CPU preparation leaves packed weights
unchanged, exactly roundtrips the swizzled FP8 scales with zero padding and
computes the expected FP32 alpha. An independent standard-library raw-range
hash and FP32 arithmetic check agrees on all16 digests and4 alphas.

Evidence: `results/qat-loader-preflight-20261112-r2.json`,
`results/qat-loader-preflight-20261112-independent-validation.json`.
The r1 report/source archive remain retained. Capability(12,0) and
flashinfer_cutlass are explicit selection stubs: no CUDA kernel, full GPU
model load, quality score, performance or final-recipe result is claimed.
The active engine diff and serial controller sources remain unchanged.


### 2026-11-12 — prevent cross-configuration quality resumes

The resume helper already required matching checkpoint/revision, engine,
packages, prompt/generation protocol and complete question/gold records, but
had not compared effective serving options. It now rejects changed options
(including KV/state precision, attention/linear backends, graph controls and
speculation), missing evidence and changed SGLANG/MAX_JOBS environment fields.
Only internal_states/startup_time/default random_seed are omitted from the
serving comparison, conditional on an integer fixed per-request seed. New
resume metadata retains the identity hashes and helper source hash. No active
evaluator, serving engine or controller source is changed.

The original measured helper is preserved at
`results/collector-sources/resume-quality-c3bf7c2b2aae5958b55ee0aab5ef29006187bfcc99d82b4e4a2c1b4e67467169.py`.
Read-only validation of the retained original and completed resumed embedding
probes finds all561 compared serving fields and5 environment fields equal;
original predictions, probe manifests and quality scores are unchanged.
`results/quality/embedding-resume-serving-identity-20261112.json` records the
configuration/evidence hashes. All98 CPU regressions pass, including changed
precision/backend/graph/speculation/environment rejection cases:
`results/tests-quality-resume-identity-20261112.txt`.


### 2026-11-12 — read-only Windows adapter mapping and shared-memory observations

The existing Windows process-memory probe and the new bounded DXGI-only
`scripts/windows-gpu-adapters.ps1` create no D3D device/resources and change
no settings. They directly map current counter LUID12BBB to the5090 D v2.
The adapter aggregate reports23.434GiB dedicated and4.151GiB shared usage;
its basic reported capacity/commit bounds pass. A vmwp instance reports
3.998GiB shared usage. Its association with a particular Linux workload is
not established. Firefox and DWM process rows violate size/commit bounds;
per-process values must not be summed into a physical-memory total.

The process and adapter samples are at distinct08:28/08:29 Singapore times.
The older baseline's128MiB vmwp shared row is a different boot/configuration,
not a controlled counterfactual. Shared allocation does not identify CUDA
spill, establish the slowdown's cause or certify fallback policy/residency.
The current quality cohort remains uncertified as a pure-physical-VRAM point.
Raw evidence and digests:
`results/progress-checks/kv-nvfp4-windows-memory-20261112.json`,
`results/progress-checks/kv-nvfp4-windows-adapters-20261112.json`,
`results/progress-checks/kv-nvfp4-windows-memory-analysis-20261112.json`.
The actual serial evaluator continues without new API/GPU requests or signals.


### 2026-11-12 — complete independent quality audits queued and exercised

`bench/audit_quality_run.py` joins the existing full1319/16k raw verifier,
fresh official MathEvaluator re-extraction/rescoring, matched GSM8K gate and
complete auxiliary comparison without generation. It requires a fresh output,
a terminal probe, exact source/evidence hashes and1319 unique official-rescore
rows with matching extractions/labels. It permits only the declared low-effort
protocol change under the explicit separate flag, with no matched-quality or
auxiliary claim. A completed evidence audit is distinct from passing the
GSM8K quality gate; no final recipe is promoted.

The completed embedding run passes the actual end-to-end CPU audit at
`results/quality/audits/embedding-full-resumed-20261112-r1/audit.json`, retaining
1281/1319 and its original auxiliary data. The source/raw results are untouched.
All101 CPU regressions pass, including stale hashes, green-but-incomplete
reports, duplicate IDs, changed extractions and output-preservation checks:
`results/tests-quality-audit-20261112.txt`.

The still-unstarted numerical child now invokes these CPU audits after all its
GPU jobs for KV, QAT, FP32-state, low-effort and selected numerical full-quality
tags. Missing/partial/mismatched evidence remains failed. Its previous source
is archived at
`results/driver-sources/archive/numerical-backend-followups-18e4557d003f691f907a5e749c276ac1e7c96360e5815bcca17e53169d4c6ba0.py`.
The active parent/post-grid/model sources and GPU experiment order are unchanged.


The actual negative-control end-to-end audit also completes at
`results/quality/audits/mixed-control-full-r1/audit.json`: official rescoring
and complete auxiliary validation pass, while the matched GSM8K result
1278/1319 retains its0.530705pp decline and gsm8k_gate_passed=false. Both
real audit reports bind their official verifier source and1319 checked rows
to the raw evidence; neither promotes a recipe.


### 2026-11-12 — NVFP4 persistent-workspace hypothesis rejected by source/meta accounting

The actual fixed-source allocator on meta tensors confirms a single FP8 K/V
workspace across all16 full-attention layers. At75,520 pool tokens plus64
padding rows, packed KV is1,238,368,256 bytes, linear scales154,796,032 and
shared workspace154,796,032 (147.625MiB), total1,547,960,320. The actual
20,480-byte compute_cell_size exactly matches that constructor's buffer sum;
the serving PoolConfigurator calls that configured method for its budget.
Global FP32 scales add128 bytes separately. The previous same-budget1.78
estimate counted only18,432 packed/scaled bytes; including the2,048 shared
workspace gives a nominal1.6 ratio to FP8. PLAN's optimization targets are
unchanged. Existing runtime capacity observations are not rewritten.

`results/memory/kv-nvfp4-full-storage-accounting-20261112.json` retains source
hashes and a frozen original probe snapshot. This is metadata-only arithmetic,
with no real allocation/GPU work. It falsifies persistent per-layer workspace
duplication and omission in cell sizing; it does not establish allocator or
transient peak memory, physical residency, paging cause or speed improvement.
No new engine patch is justified by this rejected hypothesis.


### 2026-11-12 — post-restart fallback-policy read and official scope review

The existing script's default read-only NVAPI query still returns−160 for
fallback setting0x10ECECC9 name/current-global/base reads; the public
power-management name control returns0. No setter/save/restart is executed.
`results/windows-fallback-read-20261112.json` retains the command, script hash
and raw response.

NVIDIA Support's System Memory Fallback for Stable Diffusion article documents
a Windows Python executable profile plus application restart. The reviewed
CUDA on WSL13.5 guide does not document this setting's WSL application scope.
Those pages establish neither WSL enforcement nor non-support. The local
setting-not-found status is not a default/no-fallback value. Policy/application
scope, fresh-process effects and physical residency remain open; the running
quality cohort and original PLAN targets are unchanged.
`results/windows-fallback-documentation-review-20261112.json` records both
primary URLs, their limited conclusions and local evidence hashes.


### 2026-11-12 — isolated attention KV-tile experiment added to pending replay

The fixed SM120/head-dim256 unified dispatcher uses BLOCK_N64; grouped
decode uses32. After reproducing all original saved outputs, the future
isolated replay now also runs the same native unified kernel with only its
KV tile set to32. It records all six original/selected launch fields and
requires exactly one native-selector invocation. Data, masks, head/query tiles
and warp count are unchanged. Root/all-row comparisons use actual saved
control and decode outputs, and any positive restoration flag also requires
same visible inputs and batched/single decode parity.

A real CPU hook/launcher fixture verifies only BLOCK_N changes, the mask is
preserved, ordinary dispatch returns to its original configuration, and the
override is restored after an injected launch error. All102 CPU regressions
pass: `results/tests-attention-tile-replay-20261112.txt`. No CUDA kernel or
serving-source change occurs. The prior unexecuted replay source is archived at
`results/driver-sources/archive/replay-attention-case-e3cdafad0573bf79f2f5437228e9fc114b7a5b9f7c2a2258695a8ae754efacf5.py`.
The existing future replay command picks up the helper at its normal idle
boundary; live parent/post-grid sources and GPU task order remain unchanged.
Actual numerical/parity/timing evidence is still pending, and no new engine
patch is justified before it exists.

### 2026-11-12 — NVFP4 prefill/decode source-route audit

The captured full-quality configuration uses FlashInfer prefill and TRT-LLM
MHA decode. Five hashed source files bind the inspected functions to the
unchanged engine diff. On SM120 the selected XQA decoder reads packed NVFP4
K/V and linear scales directly; its inspected decode functions do not invoke
the FlashInfer FP8 workspace preparation path. FlashInfer prefill gathers the
actual cached prefix, then materializes BF16 K/V and converts them to FP8.
This narrows source hypotheses about transient allocations without measuring
an allocation peak, kernel time, spill or cause of the live slowdown.

Evidence: `results/memory/kv-nvfp4-route-source-audit-20261112.json`; explanation
and limitations: `docs/source-audit.md`. Source hashes and report link were
independently reread after writing. No GPU work, serving-source change or
active-job interruption occurred; candidate quality remains incomplete.

### 2026-11-12 — retained capped NVFP4-KV quality response

At the normal completion of the monitored long request, `gsm8k-255` contains
16,384 generated/reasoning tokens, zero answer tokens, `finish_reason=length`,
and upstream `symbolic_correct=false`. Its recorded generation interval is
2,606.442s (43.441min). The continuing full evaluation advanced beyond255 rows;
the same controller/scheduler/evaluator identities remained live throughout
the bounded host observation. The original1319 examples and16k cap are intact.

`results/progress-checks/kv-nvfp4-long-response-20261112-r2.json` binds the raw
line number/hash, complete response metadata and text hashes. A separate read
verified those hashes/fields against the retained raw line. This is one actual
truncated response, not an aggregate quality result, a demonstrated20–50k loop,
or evidence of the cause of slow execution. No model interruption or settings
change occurred. Full quality and independent official rescoring remain open.

### 2026-11-12 — second monitored NVFP4-KV response reaches the fixed cap

`gsm8k-340` completed with16,384 reasoning/generated tokens, zero answer tokens,
`finish_reason=length` and upstream `symbolic_correct=false`. Its recorded
generation interval is2,590.531s (43.176min). The continuing evaluation advanced
beyond340 rows and began subsequent requests under the same live process
identities. The full1319/16k protocol and serial GPU ownership are unchanged.

`results/progress-checks/kv-nvfp4-long-response-20261112-r3.json` binds the raw
line number/hash, complete response fields and text hashes. A separate reread
verified those fields/hashes and elapsed interval. This retains another actual
capped response; aggregate quality, official full rescoring, causality and the
reported20–50k-token loop remain unproven. No model interruption, setting change
or concurrent GPU experiment occurred.


## 2026-11-12 — completed user-authorized three-hour reduction

The old full queue is cancelled with434 unchanged raw rows (433 responses and
one request error); partial official rescoring is418/433 versus424/433 on the
same baseline IDs. All10 reduced trials below finish, with320 new thinking
examples and a complete independent audit. The final host has no model process
and port30000 is released. Full original PLAN certification is not inferred.

| Configuration | Execution /minutes | Output tok/s,1k /8k | Thinking | Default greedy /PPL | Strict capture |
|---|---|---|---|---|---|
| Original-weight control before |passed /3.41|59.12 /56.29|Not rerun|Not rerun|Identical|
| Balanced embedding |passed /19.33|64.59 /59.38|62/64|32/32 /6.93278|Identical|
| Maximum-context embedding |passed /20.91|68.20 /62.53|62/64|32/32 /6.93278|Identical|
| QAT weights |passed /15.68|55.89 /53.00|62/64|32/32 /6.73671|Different|
| FP32 recurrent state |passed /14.07|60.87 /56.31|63/64|31/32 /6.95788|Different|
| cuDNN FP4 |passed /4.02|61.93 /59.52|Not rerun|Not rerun|Different|
| FlashInfer GDN decode |passed /3.10|71.50 /67.49|Not rerun|Not rerun|Different|
| FlashInfer GDN prefill |passed /3.34|69.89 /66.13|Not rerun|Not rerun|Different|
| Low-effort quality / default serving |passed /10.55|70.36 /67.03|64/64|32/32 /6.95788|Identical|
| Original-weight control after |passed /2.97|69.73 /66.13|Not rerun|Not rerun|Identical|

Controls drift17.95%/17.49% before→after, with a further
identical default-request control at70.36/67.03tok/s in the low-effort trial.
No causal speedup or significance is claimed. Low-effort changes only its
thinking requests:64/64 and19,006 generated tokens versus63/64 and26,462 on
the same baseline IDs (−28.18%); this is a separate sampled protocol.

Balanced/max-context scripts now match passed flags and logprob defaults.
Verified real inputs are48,640/81,400 tokens, with303.7s/312.9s short text soaks.
Both complete62/64 thinking and32/32 greedy; existing embedding full quality
1281/1319 remains a separately audited result. Final-profile1319/three-repeat/
7200s gates and the physical/no-desktop cohort are not established.

Actual attention capture/native replay is complete. Tile32 matches all24,576
candidate decode values, while the full NGRAM strict test still matches0/8.
Patch0006 stays default-off. Six patches reconstruct eleven engine files exactly.
The ≥1.5× lossless speed and matched physical2× context targets remain unmet
or unproven.

Evidence: [delivery](../docs/three-hour-delivery.md),
[raw-bound final audit](budget-3h-20261112/final-audit.json),
[independent final verification](budget-3h-20261112/final-verification.json).
