# GDN decode/verify precision investigation

The four-draft NGRAM run passes capacity but differs from the original greedy
control on every real prompt. Per-request accepted length is about1.13–1.58.
Evidence: `results/validation/ngram-d4-slots4/token-differences.json` and the raw
probe. This makes lossless verification a concrete remaining requirement.

Two arithmetic differences appear in the fixed source:

- Packed GDN decode in `kernels/ops/attention/fla/fused_recurrent.py` forms beta
  with sigmoid in FP32, rounds it to the input gate dtype, then returns to FP32.
  Its real gate input is BF16. Generic verification in
  `kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py` retains the
  FP32 sigmoid result. These are different recurrences even with identical inputs.
- Packed decode persists state after each token in the pool dtype. The generic
  chain verifier retains FP32 state between steps; its explicit BF16 rounding
  option is tied to the KDA CuTe-matching branch. The tree path reloads a parent's
  typed intermediate snapshot, so chain/tree cases must be distinguished.

Normalization, reduction layout, GEMMs, convolution and sampling can introduce
additional differences. The source findings do not yet establish their relative
contribution to the observed full-model mismatches.

`bench/gdn_verify_parity.py` isolates recurrence with this model's16 key heads,
48 value heads and128-dimensional state. It compares actual packed decode,
actual chain/tree verification and an independent PyTorch recurrence, using
identical inputs across gate/state dtype cases. It checks persistent-state
isolation, finite outputs, reference/layout agreement and exact bit differences.
The16 cases include the same linear ancestry through both chain and tree paths,
so typed-parent reload is tested without changing inputs or ancestry.
The legacy GPU diagnostic completed all16 cases in
`results/validation/gdn-verify-parity-legacy.json`. With FP32 gates and FP32
state, all modes have bit-identical output/state. FP32 gates with BF16 state
match in one-token and tree modes, while chain verification differs because
it does not persist BF16 state each step. BF16 gates differ even for one token.
This isolates the casting/state-persistence effects at the actual model geometry;
it does not establish that other full-model operators are identical.

Opt-in `SGLANG_OPT_GDN_VERIFY_MATCH_PACKED=1`, delivered as
`patches/0003-gdn-verify-packed-precision.patch`, applies beta casting and
per-step state rounding only to GDN target verification. The packed decode path
is unchanged. The flag defaults off and rejects unsupported KDA/gating settings.
`bench/gdn_verify_parity.py --match-packed-gdn-decode` passes all16 cases with
bit-identical output/state (`results/validation/gdn-verify-parity-packed.json`).
The complete41-test CPU regression suite also passes.

The subsequent actual NGRAMD4/pin4 trial passes capacity but still matches0/8
greedy prompts (`results/experiments/ngram-packed-verify-r1/tokens.json`). Its
pipeline stops at that required gate before speed testing. Thus this patch fixes
a demonstrated isolated recurrence discrepancy, but does not satisfy full-model
losslessness. Remaining projection/convolution/attention and prefill/verify
numerics require separate localization; no correctness or speed promotion.

Although recurrence is only about2.2% of summed baseline GPU time, preserving
its decode semantics may be necessary to use speculative decoding correctly.
That is the rationale for this correctness experiment; no speed gain is claimed.

The subsequent convolution audit found a separate arithmetic difference: the
tree verifier accumulates newest-to-oldest, while sequential decode accumulates
oldest-to-newest. A finite BF16 cancellation example with history
`[1, 2^-24, -1]`, current input0 and unit weights produces0 in sequential order
and2^-24 in reverse order; the difference survives SiLU and BF16 output storage.
This demonstrates an operator discrepancy rather than its frequency in inference.

`bench/gdn_conv_parity.py` exercises the actual CUDA kernels at the model's
10240-channel geometry: two state layouts, bias/no bias, one-token, chain,
tree-chain and branching-tree ancestry. Legacy output is exact in8/16 cases;
all16 intermediate states are exact. Default-off
`SGLANG_OPT_GDN_VERIFY_CONV_ORDER=1` in patch0005 sums tree products in sequential
order and passes all16 output/state comparisons. Default-off results after the
patch reproduce the legacy outputs exactly. The opt-in guard requires CUDA,
BF16, width4 and non-circular state; existing default CPU/NPU call arguments
are unchanged. Raw cases and hashes are retained in
`results/validation/gdn-conv-correction-summary.json`.

Actual NGRAMD4/pin4 with0003+0005 passes capacity but still matches0/8 original
greedy prompts (`results/experiments/ngram-packed-conv-gate/`). Seven outputs
equal the earlier recurrence-only capture and one differs. This does not locate
the remaining full-model cause or establish representative error rates. No
speed test is admitted. The full47-test CPU regression suite passes, including
default-off behavior and negative soak-evidence checks.

The next row-batching diagnostic uses actual target checkpoint weights and the
native `ModelOptFp4LinearMethod.create_weights`, weight preparation and `apply`.
Eight representative ordinary linears cover GDN QKV/Z/output components,
concatenated MLP gate/up, MLP down, fused attention QKV/output and the vocabulary
head. With synthetic BF16 activations at two magnitudes, row counts1/2/4/8/32,
and ordinary/ATen-invariant modes, all160 cases have bit-exact packed activations,
logical activation scales and outputs versus sequential single-row execution.
The scale-layout inverse is checked against the actual unswizzled quantizer.
Diagnostic wall time3.86s, peak allocated996,719,616B. This does not cover the runtime's
fused GDN QKVZ projection, fused SwiGLU/proj-conv paths, full-model activations or
inference-error frequency. It provides no evidence for an FP4 padding patch.

Artifacts: `bench/fp4_row_parity.py`,
`results/validation/fp4-row-parity-r1.json` and
`results/fp4-row-boundary-guard.json`. The guard verifies no model server remains,
runs the isolated diagnostic, then resumes the same scan PID/start time; no
engine code changes were made. Deterministic inference's ATen registrations do
not themselves replace this custom FP4 operator. The unquantized BF16 GDN gate
projection is a separate path, now independently checked through the actual
`UnquantizedLinearMethod` with `auto` backend (the SM120 native F.linear path).
The diagnostic loads fused `[b;a]` weights from layers0/1/30/62 and tests two
activation magnitudes and M1/2/4/8/32/128/1024 against sequential single-row
execution. Native mode is exact in14/56 cases, with maximum absolute output
difference0.0625 across the synthetic cases. At the actual four-row verification
shape only2/8 cases are exact (maximum difference0.0009765625). The upstream
ATen invariant mode is exact in all56 cases, with finite outputs throughout.

`results/validation/gdn-gate-projection-parity-r1.json` retains all112 cases,
native method/diagnostic hashes and checkpoint revision. Independent coverage
and finite/exact checks are in `results/validation/projection-parity-summary.json`.
The clean-boundary guard again resumes the same scan without engine changes.
This identifies a concrete additional operator difference and supports trying
the already queued matched deterministic controls. It does not establish how
often this difference changes model outputs or prove full-model speculative
equality. Existing upstream invariant operations resolve the isolated case;
no new BF16 GEMM patch or speed benefit is claimed.

## Actual-forward localization prepared (2026-11-11)

The eager control/NGRAM gate is queued behind the uninterrupted runtime test.
If strict equality remains negative, `bench/trace_pair.py` is ready to locate
the first observed dense module/state difference using the actual model and
prompt, rather than another unrelated synthetic kernel comparison. No GPU
trace result exists yet.

`EagerRunner._execute_decode` and `_execute_extend` call the outer
`model.forward` directly; outer PyTorch hooks therefore do not fire. The public
`--forward-hooks` target must be `model` (the actual Qwen backbone), not the
empty root name or the checkpoint weight prefix `model.language_model`. The
backbone is `Qwen3_5ForCausalLM`; its named layer attributes are `layers.N`.
The factory `forward_trace:make_hook` installs bounded hooks during an unarmed
capacity request. Both graph phases must be disabled because hook registration
happens after graph capture and replay does not execute Python hooks.

The client flushes the cache, arms one actual first real prompt, and requests
eight greedy tokens. The collector records one full prefill and its first
decode/verify, actual IDs and positions, the last prefill/first verify row at
observed module boundaries, GDN persistent state after prefill and before the
first step, and final backbone hidden states. It never compares verify's
unmodified persistent post-state with sequential decode's updated post-state.
CPU copies synchronize execution; these traces cannot support timing claims.
Each worker has a256MiB bound and each tensor artifact has a SHA256 digest.
Opaque quantized layouts, missing/fused module boundaries and nonfinite or
incomplete traces cannot silently become equality evidence.

At a clean GPU boundary only:

```bash
source scripts/runtime-env.sh
python bench/trace_pair.py --output results/forward-traces/eager-ngram-r1
```

The driver refuses to compete with an existing model process. It has not been
started during the mixed soak. Actual probe-built CLI checks and factory imports
are retained in `results/forward-trace-cli-validation-20261111.json`. Four CPU
regressions use the real SGLang hook manager and real PyTorch forwards, including
direct outer `.forward` and keyword backbone arguments, state slot selection,
subsequent state mutation, size exhaustion, tampered files and incomplete traces.
The full suite is77 passing checks
(`results/tests-forward-trace-final-20261111.txt`); this is tooling verification,
not full-model numerical proof.

## Eager strict results and observed request metrics

The eager control completed six formal serving repeats at13.875/13.615tok/s
for1k/8k. Eager EAGLE and NGRAM each passed24,576-token capacity, then
matched0/8 own-control greedy outputs. The paired GPU trace is waiting for the
validation queue to exit naturally.

| Trial | Exact outputs /8 | Weighted accepted length on eight real prompts | Median reported real-request decode tok/s | Reported synthetic8192/1024 decode tok/s |
|---|---|---|---|---|
| Eager EAGLE |0|2.983|34.123|46.480|
| Eager NGRAM |0|1.316|15.057|46.507|
| Native EAGLE |0|2.821|124.993|169.947|
| Native packed EAGLE |0|2.651|122.582|167.311|
| Native packed EAGLED2 |0|1.829|85.150|89.669|
| Deterministic embedding DFlash |0|3.857|75.624|138.866|

The real-request column comes from each retained512-token greedy gate capture.
Weighted accepted length uses total generated tokens divided by verification
count. Capacity requests use synthetic IDs and have their own column. Source
hashes, full per-prompt values, own controls and independent token-count checks
are in `results/speculative-eager-native-observations-20261111.json`. Formal
serving and quality comparisons still follow the required gates.
