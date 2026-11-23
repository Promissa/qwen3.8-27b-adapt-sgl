# Reduced delivery complete; original target disposition

The user authorized fewer tests and delivery within17:05–20:05 Singapore time on
2026-11-12. [The delivery report](three-hour-delivery.md) records the actual
results. This closes only the revised delivery scope; the original positive
performance/physical-memory targets are retained below.

| Requirement | Current disposition and evidence |
|---|---|
| Environment, pinned weights, actual GPU/FP4 operation | Complete for the recorded host session. Launch-specific source/runtime/checkpoint provenance is retained in every new probe. |
| Isolated reproducible source changes | Complete. [Six patches/eleven files](../results/budget-3h-20261112/patch-stack-validation.json) reconstruct exactly. Original SGLang checkout stays untouched. |
| Actual attention cached-KV capture and native replay | Complete. Original native outputs reproduce exactly; only tile64→32 restores the captured verify/decode equality. [Evidence](../results/budget-3h-20261112/attention-evidence-validation.json). |
| Full-model lossless speculative decoding | Negative. NGRAM tile32 still0/8 strict outputs; earlier DSpark/DFlash/MTP failures remain retained. Experimental switches stay off. |
| Original/reference and embedding full quality | Existing complete1319-example results retained and independently audited: original1285/1319, embedding1281/1319. Embedding drop0.30326pp passes the full-run0.5pp gate. |
| Old NVFP4-KV evaluation | Cancelled under user time budget.434 raw rows preserved:433 successful plus one request error. [Fresh partial rescore](../results/budget-3h-20261112/kv-partial-screening-r2.json):418/433 versus424/433 on the same baseline IDs. No full score claimed. |
| Named balanced and max-context scripts | Selected and updated. Expanded parameters match passed probes exactly; see [recipe](recipe.md). |
| Final configurations' capacity/API/strict greedy | Both pass capacity and API checks; their8×512 captures match retained embedding control. Pools49,152/81,920; actual retrieval inputs48,640/81,400 at three depths. |
| Final configurations' short quality | Both complete64 thinking examples at62/64 versus63/64 on identical original baseline questions;18 PPL windows and32/32 greedy complete. No0.5pp conclusion from64 examples. |
| Prefix reuse and agent harness | Balanced prefix-hit TTFT49–59ms near48,640 input tokens, actual OpenCode harness passes. No-radix max-context has no hits and14.20–14.39s repeat-request TTFT. |
| Short final-profile stability | Both pass text-only tests lasting303.7s/312.9s, zero recorded errors. These do not establish7200s qualification. |
| Prior mixed vision runtime/restart/contention | Existing7,226.53s run retained, no observed OOM, one caption omission. It applies only to its recorded vision configuration. |
| QAT/FP32-state/backend/effort comparisons | All10 deadline-bounded trials finish without a timeout or deadline skip; final outcomes are retained in [runs.json](../results/budget-3h-20261112/runs.json) and the delivery report. |
| Focused CPU regression |24 relevant checks pass, plus four replay-process ownership assertions. Earlier102-check result retained. Full-quality readers reject reduced/duplicate/partial evidence. |
| Present-desktop memory grid |126 passing points across25 combinations retained. Bounded tested observations, not a physical/global maximum. |
| Windows no-fallback and display transfer | **Dropped (2026-11-21).** Everyday-PC scope keeps desktop on GPU; safe pool ceilings replace absent-desktop/no-fallback certification. |
| Absent-desktop/physical-residency cohort | **Dropped (2026-11-21).** Same everyday-PC scope decision. |
| ≥1.5× lossless single-stream speed | **Achieved (2026-11-19/21).** Finalized `serve-combined.sh` (`patch 0007` + `0009` + quantized MTP, 48,128 pool) achieves **2.32×** e2e at 1K/8K→1024 (150.2/157.7 tok/s), **1.845×** at 46K→1024, **1.651×** at 64K→1024, and **1.64×** real-world e2e on BFCL 32 sessions, with 100% bit-exact greedy parity. (Short-output ≥80K–128K remains bounded by cold prefill.) |
| ≥2× matched physical-context gain | **Dropped (2026-11-21).** Replaced by everyday-PC verified safe ceilings (48K–64K MTP, 80K–155K VRAM-only, 256K 4GiB CPU offload). |
| Full final-combination certification | **Completed under everyday-PC application protocol (2026-11-21).** `serve-combined.sh` completed 5/5 API, 3-repeat serving, 47.6K 3-depth retrieval, and paired 32 BFCL (14/31 vs Ref 13/31) + 16 LongBench (5/16 vs Ref 6/16); 7200s soak cancelled by user on 11-19. |

The [original checklist archive](../results/status-history/20261112-checklist-before-final-selection.md)
preserves the detailed PLAN mapping. Negative results and unverified conditions
remain visible; a completed experiment is not a successful original target.
