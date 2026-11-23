# First actual decode/verify difference — 2026-11-12

## Completed cached-KV replay and full strict check

During the user-authorized3h completion window, both cached-KV capture probes
and the native replay completed. All four original outputs reproduce exactly:
control/verify prefill plus the first control decode and NGRAM verify. Their
first query, visible cached K/V and masks match. Batched decode replay also
matches independent single-query decode replay for all24,576 candidate values.

The native unified kernel uses KV tile64 in the captured SM120/head-dim256
case. Overriding only that tile to32 (preserving head/query tiles and eight
warps) makes all candidate outputs bit-identical to decode replay, including
the control root. The original verify differs in10,990/24,576 values,
max absolute difference0.0625. This isolates the numerical route/tile effect
for this actual case.

Patch0006 adds the default-off `SGLANG_OPT_MHA_VERIFY_TILE32` switch, restricted
to target verification with SM120 FP8 K/V and head dimension256. The subsequent
real8-prompt/512-token NGRAM strict test still matches **0/8** complete outputs.
First differing generated-token indices move from6–42 to38–256; every matching
prefix becomes longer. Seven first differences occur above256 total input plus
output tokens, but one occurs at156, so a universal split-boundary cause is not
established. The patch is experimental and excluded from lossless serving.

Evidence: [capture](../results/forward-traces/eager-ngram-attention-kv-r1/pair.json),
[native replay](../results/forward-traces/eager-ngram-attention-kv-r1/kernel-replay-budget-r2.json),
[independent identity checks](../results/budget-3h-20261112/attention-evidence-validation.json),
[full strict comparison](../results/budget-3h-20261112/ngram-tile-strict-comparison.json),
[six-patch reconstruction](../results/budget-3h-20261112/patch-stack-validation.json).

The initial replay attempt falsely matched its timeout supervisor as another
replay owner. The guard now matches actual Python replay processes; the first
failed report is preserved and a fresh output contains the successful replay.

## Earlier localization and preparation record

The sections below retain the earlier preparation state. Their queued/pending
language is superseded by the completed results above; the exact earlier file
is archived in `results/status-history/20261112-attention-before-budget.md`.

The paired eager Triton trace first differs at the output of
`model.layers.3.attn` (layer index3, the first full-attention layer). The actual
Q/K/V inputs there are bit-identical. The93 earlier shared dense boundaries
in the first-step comparison are also identical. This narrows the next
investigation to full attention; it does not prove a particular kernel or
cache operation is the cause.

[Raw comparison](../results/forward-traces/eager-ngram-r1/comparison.json) ·
[independent validation](../results/forward-traces/eager-ngram-r1/independent-validation-20261112.json).

| Observation | Actual evidence |
|---|---|
| Inputs | Same48 native prompt IDs and positions, independently checked against each owned server tokenizer; same first step input ID and position |
| Modes | Control has one decode query; NGRAM verifies four query rows, comparing its root row with control |
| Execution | Both use deterministic Triton attention; `--cuda-graph-config` explicitly disables prefill and decode. Actual traced forwards report CUDA graph false |
| Prefill | All1780 shared dense boundaries match bit-for-bit; the collector retains the last prompt row plus whole post-prefill GDN states |
| First step | All96 shared persistent-before-step GDN state tensors match; the first93 shared dense boundaries match |
| First attention inputs | Query, key and value inputs at layer3 match bit-for-bit |
| First attention output |2962/6144 values differ; maximum absolute difference0.0625 |
| Capture size |228,578,320 control bytes and229,757,968 NGRAM bytes; both below the256MiB bound |
| Missing structure |96 control-only and48 candidate-only inner GDN leaves reflect tuple versus tensor returns; shared outer boundaries still compare. Each frame also explicitly excludes161 opaque layouts |

The full-prefill KV cache and custom attention mask are not part of this bounded
capture. The prefill result is therefore not a claim that every cached KV row
was independently checked. The actual trace contains eight output tokens for
one prompt, not a replacement for the retained eight-prompt/512-token strict
failures. CPU copies synchronize this diagnostic; its timings are not serving
benchmarks.

## Fixed-source attention dispatch

The measured command enables deterministic inference. In this exact source,
[`TritonAttnBackend.forward_extend`](../engine/sglang/python/sglang/srt/layers/attention/triton_backend.py)
stores current KV and routes deterministic extend/verify through
`_forward_extend_unified`. It combines prefix and extend locations, then reads
both from the cache through the one-stage unified kernel.
`forward_decode` stores current KV and invokes the standard two-stage decode
kernel. Deterministic mode disables the separate Lean choice.

Both the grouped decode and unified extend kernels downcast Q to the FP8 cache
dtype on this NVIDIA path. Merely assuming one uses BF16 Q and the other FP8 Q
would misidentify the source. The dispatch, reduction shape, custom mask and
cache-address metadata need an isolated replay using identical cached KV.
Primary local sources:
[decode_attention.py](../engine/sglang/python/sglang/kernels/ops/attention/decode_attention.py),
[extend_attention.py](../engine/sglang/python/sglang/kernels/ops/attention/extend_attention.py).

The next GPU experiment should retain the first attention layer's complete
visible KV/mask metadata and replay decode versus unified verify on identical
data. A decode-equivalent verify path is a candidate only after that isolation;
no new engine patch or losslessness improvement is claimed here. Any fix still
needs isolated parity, the complete strict prompt gate, matched quality and
formal serving benchmarks before final-combination validation.

## Reusing the immutable diagnostic

Independent raw-tensor recomputation exactly reproduces the serialized report.
The original reuse check compared Python tuple signatures with JSON lists and
would falsely reject it. The reader now compares their canonical JSON form.
Its provenance check still requires unchanged collector, capture-client and
comparison sources, unchanged serving source and revalidated raw tensor hashes.
A corrected read-only runner requires the exact archived measured runner; both
runner hashes are reported. The measurement manifest and tensor files were
not rewritten.

[Read-only reuse proof](../results/forward-traces/eager-ngram-r1/reuse-verification-20261112.json).
The source archive is
[the original measured runner](../results/driver-sources/archive/trace-pair-4c00802ee379114ccaaa070fa3cb5a77a6844e14bdfb7f52f4047b2b937ccf57.py).
All102 CPU regressions pass, including rejection of changed captures, missing or
corrupted archives and altered comparisons. The real retained-data CLI reuse
also passes while the separate serial quality queue remains active.

## Prepared cached-KV collection

`bench/attention_case.py` is an extra public backbone hook beside the original
collector. It records complete selected-layer Q/K/V/output and only the actual
logical KV rows, preserving raw cached bytes, original dtype/storage geometry,
mask rails and decode split metadata. It supports C1 page-size1 ordinary MHA,
rejecting unsupported MLA/DCP/window layouts. The total tensor bound is32MiB;
physical pool rows are gathered on-device before CPU copies. It does not dump a
whole allocated cache. A mask mismatch is reported as diagnostic evidence rather
than silently corrected.

At a clean boundary the prepared command is:

```bash
.venv/bin/python bench/trace_pair.py --attention-cases \
  --output results/forward-traces/eager-ngram-attention-kv-r1
```

This is queued in the not-yet-started numerical child after the current
post-grid quality queue completes. The parent PID/start, terminal predecessor,
absence of model processes and available port are checked. The old future
script and prior read-only runner are archived with their original hashes;
original serving and live post-grid sources remain unchanged.
[Queue handoff](../results/attention-case-queue-handoff-20261112.json),
[concrete plans](../results/attention-case-plan-20261112.json).

Actual PyTorch/public-hook tests cover exclusion of unarmed warmups, noncontiguous
physical slots, complete prefix/current cached bytes, copy independence after
later cache mutation, reported mask differences, corrupt data and byte-limit
failures. The queue boundary test rejects live models and occupied ports.
All102 CPU checks pass. Default old-trace planning and real retained-data reuse
still pass; the original three capture/comparison sources remain exact.
[Latest checks](../results/tests-attention-tile-replay-20261112.txt),
[old-data compatibility](../results/forward-traces/eager-ngram-r1/reuse-verification-attention-plan-20261112.json).
The additional GPU cases and isolated decode/unified-kernel replay are pending.

## Prepared isolated replay

The serial child now invokes `bench/replay_attention_case.py` immediately after
a successful additional capture, with no model process and a free port. A
failed capture skips replay; a failed isolation remains recorded and does not
promote any engine change. The former queued script is archived by hash.

The replay reconstructs the original physical cache addresses, dtype and
strides, zeroing uncaptured slots. It bounds reconstruction and temporary
scratch at256MiB each and rejects contradictory duplicate physical bytes,
invalid addresses or unsupported layouts. It requires every saved prefill and
first-step output to reproduce bit-for-bit through its original kernel before
attempting decode-route attribution. The alternate replay uses one KV rail per
actual query mask, including tree branches; position and deterministic split
checks must pass. Batched results are compared with individually replayed
queries and the original control root. Output allocations are independent.

CPU hook/reconstruction tests use actual PyTorch FP8 byte views, noncontiguous
query layouts and branched masks. All96 checks pass. The GPU report will be
`results/forward-traces/eager-ngram-attention-kv-r1/kernel-replay.json`;
it does not exist yet. This is a queued diagnostic, with no timing, new patch
or full-model losslessness result.

## Prepared KV-tile isolation

The fixed CUDA source selects unified `BLOCK_N=64` for this256-dimensional
Qwen head on SM120, while grouped decode streams32 keys per block. This is a
source-level candidate explanation for a different online-softmax accumulation
path, not a measured explanation of the retained first difference.

After every original output reproduces exactly, the pending replay additionally
launches the same native unified kernel with only `BLOCK_N=32`. Head tiles,
query tile, warp count, data, scales and actual mask stay unchanged. The report
records both complete configurations, checks that the native selector was used
exactly once and compares all query outputs with the decode replay and the
actual control root. Any positive flag additionally requires same visible
inputs and batched/single decode parity. The temporary selector override is
restored on success and failure; no serving source is edited.

The CPU launcher test verifies parameter isolation, unchanged mask forwarding
and restoration after a launch error. All102 checks pass; this fixture executes
no CUDA kernel. The actual tile experiment remains queued inside the same
future serial replay, with no numeric parity, timing or new engine patch result.
[Queue handoff](../results/attention-case-queue-handoff-20261112.json).
