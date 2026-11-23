# Three-hour completion window

User instruction on2026-11-12: reduce test quantities, estimate time from earlier
runs, and finish the remaining work within3h. The budgeted window is
**17:05–20:05 Singapore time**. GPU jobs stop by19:35, reserving30min for the
final evidence audit and delivery.

**Revised delivery completed at2026-11-12 19:06:13 Singapore time,
121.0min after the request, within3h.** All10 planned reduced trials
finish successfully as executions;320 new thinking examples are complete.
The final independent audit has no missing trials. No model process remains,
port30000 is released, and the host GPU is accessible. Original positive targets
are not declared achieved by finishing this reduced scope.

The superseded full-evaluation queue is stopped. **434 raw NVFP4-KV predictions
are preserved byte-for-byte**:433 successful responses and one request-error
record. They are partial evidence, not a completed1319 score. Original manifests
are archived beside the cancellation record.

## Runtime budget and changed sample counts

| Work | Earlier measurement or estimate | Current protocol |
|---|---|---|
| Remaining NVFP4-KV full run | 433 measured responses averaged79.49s; remaining estimate19.56h | Stop/preserve434 completed rows |
| Attention capture | Earlier serial pair3.5min; new pair and replay completed | Full bounded tensor/KV capture and native replay retained |
| Thinking quality | First64 questions took7.27min baseline and8.33min embedding |64 examples/configuration; same16,384 output cap and upstream scorer |
| Auxiliary quality | Existing full120-window/200-example baseline retained |6 windows/domain and32 nonthinking greedy examples |
| Serving | Existing three-repeat full baseline retained |1 request/load at1k/8k input,1024 output; no noise/significance claim |
| Final stability | Existing7,226.53s mixed runtime retained |300s/configuration; does not establish a new two-hour gate |

Short-run outputs are screened and independently recomputed, with the original
full-quality readers still rejecting reduced evidence.64 questions have a
1.5625-percentage-point score resolution, so they cannot establish a0.5pp bound.

## Completed technical work in this window

- Captured actual first-full-attention cached K/V, masks, Q and original outputs.
  All four saved prefill/decode/verify outputs reproduce bit-for-bit through
  their original native kernels. Visible inputs match between the paired runs.
- Native unified attention with only KV tile64→32 matches all24,576 candidate
  decode-output values, including the control root. Other launch fields remain
  unchanged. This is an isolated numerical result.
- Patch0006 exposes this as a default-off experimental verify switch. The real
 8-prompt/512-token strict test still matches0/8 complete outputs, although all matching prefixes
  become longer. It is excluded from recommended lossless serving.
- All six patches independently reconstruct eleven engine files exactly.
  Relevant CPU checks pass; duplicate/partial subsets remain rejected and
  reduced rows cannot enter the original1319-example quality gate.
- Current reduced baseline completed in204.68s. Its two single-request results
  are59.12/56.29tok/s at1k/8k input. These describe this cohort and protocol.
- The preserved NVFP4-KV responses were independently re-extracted/regraded by
  the official scorer on CPU:418/433 successful responses correct, versus424/433
  on the identical baseline questions, a descriptive subset drop of1.3857pp.
  The separate failed request remains explicit. Together with its retained slow
  runtime and failed prefix-TTFT results, this candidate is excluded from the
  current recommendation; no1319-example quality conclusion is claimed.

## Selected launchers and measured behavior

Both named launchers now select their tested converted-embedding configurations.
The generated effective arguments match the passed probes exactly; user overrides
remain available. Original strict greedy captures and quality checks refer to
text-only serving with experimental speculative switches off.

| Measurement | Balanced | Maximum context |
|---|---|---|
| Static memory fraction / state slots |0.88 /4|0.92 /1|
| KV token-pool capacity |49,152|81,920|
| Real near-limit input / retrieval |48,640 /3 of3 depths correct|81,400 /3 of3 depths correct|
| Prefix reuse |All3 observed hits;48.7–59.5ms TTFT|Disabled; repeat TTFT14.20–14.39s|
| Single-request output tok/s,1k /8k input |64.59 /59.38|68.20 /62.53|
| Thinking screening |62/64|62/64|
| Nonthinking greedy / reduced PPL |32/32 /6.93278|32/32 /6.93278|
| Actual text-soak duration |303.70s, passed|312.91s, passed|
| Actual OpenCode harness |Passed|Not rerun; this profile is for one-off long inputs|
|8-prompt greedy (512-token cap) versus retained embedding control |Identical|Identical|

The original baseline on the same64 thinking questions scores63/64. The
embedding checkpoint has a separately audited full result, but neither reduced run independently establishes a0.5pp final-combination bound.
The short speed figures are whole-configuration observations, not isolated
embedding attribution or statistically qualified gains. In particular, the
24,576-token control cap must not be used to manufacture a2× context claim.
Use balanced for repeated coding-agent conversations; the maximum-context
variant favors a larger one-off input at the cost of prefix reuse.

## Request-level low-effort observation

On the original-weight24,576-token control, the separate `reasoning_effort=low`
64-question run scores64/64 versus63/64 on the retained baseline's same IDs.
It generates19,006 tokens versus26,462 (−28.18%), including10,740 reasoning
tokens. This is a single sampled subset under a changed reasoning protocol,
not a full-quality guarantee or a kernel-throughput gain. The default recipes'
checkpoint is different; that low-effort combination is not claimed tested.
No default effort is changed.

## Final serial validation results

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

Strict capture compares the embedding variants with their retained embedding
control and the other variants with the retained original-weight control, on
eight fixed prompts with a512-token cap. `Different` is not a lossless result.
Low effort applies only to the thinking-quality requests in its row; its
synthetic benchmark, nonthinking greedy and PPL use default control requests.

## Performance interpretation

The before/after original-weight controls change59.12→69.73tok/s at1k and
56.29→66.13tok/s at8k: **17.95% /17.49% drift**. The low-effort
trial's default-request serving control is70.36/67.03tok/s. The fastest requested
backend point, GDN decode71.50/67.49, is only1.62% /0.69% above that control,
from one request per load. No repeatability, statistical significance or causal
throughput gain is established. The functional/context results remain measured
points; no ≥1.5× lossless speedup is claimed.

A trial marked `passed` completed its planned checks; its64-example score
does not certify the original full-quality gate.

The controller ran the jobs serially with hard allowances. All10 completed
before the GPU cutoff, without a deadline skip or timeout. It confirmed cleanup
between probes; the final host check also confirms model-process absence and a
free port. No additional C2 test was added: the retained present-desktop grid
contains prior C2 observations, while both final launchers explicitly use C1.

Windows no-fallback enforcement and the absent-desktop physical-memory cohort
remain unverified. Existing shared-memory counters do not establish CUDA spill.
Original speed/context/quality targets remain unchanged; failed or unproven
targets are stated above and in the completion disposition.

Evidence: [budget](../results/budget-3h-20261112/plan.json),
[queue cancellation](../results/budget-3h-20261112/queue-cancellation.json),
[completed runs](../results/budget-3h-20261112/runs.json),
[Attention validation](../results/budget-3h-20261112/attention-evidence-validation.json),
[strict comparison](../results/budget-3h-20261112/ngram-tile-strict-comparison.json),
[patch reconstruction](../results/budget-3h-20261112/patch-stack-validation.json).
The [partial-KV audit](../results/budget-3h-20261112/kv-partial-screening-r2.json)
retains all successful labels and the request-error identifier.

Final evidence: [combined raw-bound audit](../results/budget-3h-20261112/final-audit.json),
[independent final verification](../results/budget-3h-20261112/final-verification.json),
[host cleanup](../results/budget-3h-20261112/final-host-state.json),
[balanced command/environment](../results/budget-3h-20261112/serve-balanced-command-validation-r2.json),
[max-context command/environment](../results/budget-3h-20261112/serve-max-context-command-validation-r2.json).
Earlier complete1319-example and7,226.53s evidence is indexed in
[status](status.md); it is not relabeled as a new final-profile test.
