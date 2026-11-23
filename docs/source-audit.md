# Local source audit (2026-11-10)

Source: `/home/pr/projects/sglang`, commit
`c3c292b65ab42efd933e0c77cfb48eb69f1b81d0`. Existing tokenizer-manager modification
is preserved. Live execution later disproved one initial assumption: the model's
text-only implementation exists, but its CLI architecture allowlist omits
`Qwen3_5ForConditionalGeneration`. The isolated checkout at `engine/sglang` adds
that one entry; the original checkout is untouched. See
`patches/0001-qwen35-language-model-only.patch` and the 16 passing checks in
`results/tests-live-integration.txt`. Real text-only model execution and the first
capacity/API/retrieval/perplexity checks have passed.

## Text-only CLI gate found during phase 1

The unpatched model exits before loading weights with
`ValueError: --language-model-only does not support ['Qwen3_5ForConditionalGeneration']`.
This is recorded in `results/probes/baseline-live-088/server.log`.
`ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES` is enforced by
`model_hook.handle_language_model_only`, independently of the model implementation.

The implementation already has the necessary paths:

- `ModelConfig.is_lm_only` disables multimodal capability/MRoPE and propagates
  `hf_config.language_model_only`.
- `Qwen3_5ForConditionalGeneration` inherits `Qwen3VLForConditionalGeneration`,
  whose initializer sets `visual=None` for text-only mode.
- Qwen3.5 guards MRoPE and deepstack accesses when the visual tower is absent;
  the base forward directly calls the text model in this mode.

The patch enables only this verified architecture, preserving unknown-architecture,
encoder-conflict and PD-disaggregation rejections. It is a feasibility fix, not a
measured performance optimization. Checkpoint loading and both capacity requests
passed in `results/probes/baseline-capacity-088/probe.json`.

## Post-capture reserve

`python/sglang/srt/environ.py` sets
`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)`.
`arg_groups/overrides.py::post_capture_kv_sizing_planned` first checks that switch,
then CUDA, sharding, MLA/unified-memory, memory saver, graph coverage and model
constraints. `model_runner_components/kv_pool_runtime.py::is_post_capture_kv_active`
uses that predicate for the target worker. Therefore this branch is not active by
default in the baseline environment. A caller can override it; logs must confirm
the resolved configuration in any measured run.

When active, `compute_post_capture_kv_resize` raises hybrid-model headroom to
`pre_capture_activation_reserve_mb_of`. Its formula includes the TP/PP term:

```
512 + 1.5 * max(chunked_prefill_size, 2048) + tp_size * pp_size / 8 * 1024
```

At TP=PP=1 and chunk≤2048 this is **3712 MiB (3.625 GiB)**, rather than 3584
MiB. This is a conditional configured minimum, not an observed runtime allocation.
Neither this value nor the estimate in PLAN.md is a measured deduction from
available VRAM.

## Embedding dispatch and file format

`layers/quantization/modelopt_quant.py::ModelOptNvFp4EmbeddingMethod.embedding`
gathers packed rows, unpacks low nibble first, indexes the E2M1 magnitude table,
applies E4M3 per-group scales and the FP32 scalar, then converts to parameter dtype.
Its weight layout is `[vocab, hidden/2] U8`, `[vocab, hidden/16] E4M3`, `[1] F32`.

The embedding method is selected in `ModelOptMixedPrecisionConfig.get_quant_method`
for a `VocabParallelEmbedding` resolving to `NVFP4`. A pure `ModelOptFp4Config`
does not itself dispatch this embedding method. Consequently simply changing the
weight tensor's dtype is insufficient. The offline converter promotes flat NVFP4
metadata to an explicit mixed-precision layer map and adds the embedding entry.
Unknown/tied layouts are rejected. Tests call the actual upstream gather method
on converted fixture tensors; real checkpoint loader and quality remain unverified.

## Reused benchmark protocol

`docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` uses `sglang-oai`, random lengths,
range ratio 1 and flush-cache for speed. The local quality bridge
`python/sglang/test/sgl_eval_utils.py::run_sgl_eval` supports explicit generation
overrides, full prediction files and incomplete-result errors. It pins its scorer
via the `sglang[test]` extra to `sgl-eval==0.1.3`; that package is now installed
in the project overlay. The driver uses this existing scorer rather than
introducing a different GSM8K answer extractor.

The live converted-checkpoint test exposed a second missing connection:
`Qwen3_5ForCausalLM._build_embed_tokens` did not receive the quantization config
or module prefix. Consequently the generic NVFP4 method was never selected,
and the unquantized loader rejected width 2560 against 5120. Patch 0002 forwards
both through the backbone hook and updates the existing Qwen4 override to keep
that constructor contract. Three focused tests cover constructor dispatch, real
packed-weight loading/gather, ordinary embeddings, the PP guard and the override.
Eleven relevant upstream tests also pass. The real converted model now passes
capacity, API and near-limit retrieval; full quality remains pending.

Its 14.51GiB load-stage memory delta is 2.95GiB below the initial flat-format
control, whereas the checkpoint's exact file reduction is about 1.7GiB. Promoting
the metadata to mixed precision also changes loader dispatch, so the entire
VRAM difference cannot be attributed to packing the embedding. A metadata-only
control is still needed for that attribution. The much larger graph footprint
and unverified sysmem policy also require conservative envelope validation.

## Native reasoning effort and prompt equivalence

The pinned Qwen3.8 templates accept `low`, `medium`, and `xhigh`; omitted effort
resolves to `xhigh`. These levels steer the prompt, rather than imposing a hidden
token budget. `xhigh` adds instructions to validate assumptions and alternatives;
`low` requests brief, focused thinking; `medium` adds neither instruction.
`enable_thinking=false` closes the thinking prefix and suppresses effort steering.

The target's template differs from the reference/QAT template, and tokenizer
padding metadata differs too. Exact rendering of all1319 official math prompts
nevertheless matches across all three tokenizers in default, low, medium, xhigh
and disabled-thinking modes. Maximum prompt lengths are respectively263,251,221,
263 and223 tokens. Raw tokenizer/template hashes, pins and corpus token hashes
are in `results/gsm8k-template-parity.json`. This verifies this math protocol only;
tool histories and multimodal rendering need their own checks.

`eval_quality.py --reasoning-effort` passes steering through the upstream
generation configuration, where it remains visible in recorded template kwargs.
The live control retains its original default effort. No effort-quality or
time-saving recommendation is justified until the corresponding runs finish.

## Speculative switch applicability

Three proposed DSpark ablations do not activate a path for this model in the
fixed source. Their descriptors remain in `configs/experiments.json`, labeled
`source_inactive`; dry-run inspection works, while live drivers reject them
instead of reporting a no-op as an optimization trial.

| Switch | Actual consumer and condition | Consequence here |
|---|---|---|
| `SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN` | `FlashInferAttnBackend` capture branch requires `EAGLE_VERIFY` and topk1. The separate unmasked `DFLASH_VERIFY` branch installs its host plan without this flag | DSpark uses `DFlashVerifyInput`, so this is not its host-plan A/B switch. Retest only in a separate feasible EAGLE/MTP recipe |
| `SGLANG_OPT_FUSED_KDA_VERIFY` | Only consumed in `KDABackend`, with the fused chain-verify capability check | This model has GDN, not KDA; there is no corresponding GDN flag consumer |
| `SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR` | Consumed in `FlashInferMultiStepDraftBackend`, created through `DraftBackendFactory` by EAGLE workers | DSpark initializes the ordinary draft `TpModelWorker`/ModelRunner attention backend. Its block draft does not instantiate this multistep backend |

`--speculative-token-map` is loaded by `EAGLEWorkerV2.init_token_map`, slices the
draft head in `init_lm_head`, and maps draft indices back through `hot_token_id`.
The DSpark and DFlash worker trees contain no consumer of this map. EAGLE also
rejects reduced-vocabulary rejection sampling in its explicit vocabulary-size
guard. The Qwen MTP getter exposes raw head weights, whereas whole-module target
head sharing is disabled when a hot map is present; therefore a packed NVFP4
head needs a dedicated loader/scale compatibility check before this route can
be considered usable. No token-map runtime support or speedup is claimed.

## Independent quantized Qwen MTP draft

`TpModelWorker._init_model_config` selects `speculative_draft_model_path` for
the draft worker. `ModelConfig._config_draft_model` rewrites a Qwen3.5 conditional
draft checkpoint to `Qwen3_5ForCausalLMMTP`, with one next-token-prediction layer.
It does not require that the target weight index itself contain that MTP. The
all-linear target declares zero MTP layers; the independent reference declares
one. Eleven checked hidden/vocabulary/attention/numerical config fields agree
(`results/external-mtp-geometry.json`). Thus the previous blanket exclusion of
EAGLE for the main target was too strong: a separate MTP checkpoint is a candidate.

`_mtp_quant_config` explicitly retains mixed-precision configs containing
`mtp.*` entries. The converter adds seven `W4A16_NVFP4` entries, removes matching
MTP exclusions, preserves the backbone map, and shares global scales across
fused QKV and gate/up groups. The ordinary `nn.Linear` context projector stays
BF16. No activation calibration is implied. Exact independent artifact checks
and the actual Qwen constructor/fused-loader fixture pass; GPU work is queued.

EAGLE currently binds shared embedding/head weights in `alloc_memory_pool`,
after the draft worker allocates its pool. Whether duplicate tables constrain
this candidate's available budget must be measured before proposing a lifecycle
patch. Likewise, an embedding-quantized target cannot be assumed compatible with
raw-tensor sharing into an ordinary draft embedding. These are runtime gates,
not established defects or reasons to bypass correctness checks.

## Remaining precision and reference feasibility gates

The FP32-state, NVFP4-KV and unmodified reference checkpoints are queued after
the prefix-policy servers. FP32 changes only the recurrent-state dtype from the
baseline recipe. NVFP4 KV uses explicit FlashInfer prefill and TRT-LLM MHA decode.
The reference starts at fraction.94 with four pinned state slots, retaining the
baseline2048 prefill chunk. These are initial capacity/API/retrieval tests, not
full quality or serving-speed comparisons.

`handle_nvfp4_prefill_kv_dequant_dtype` resolves FlashInfer prefill to an FP8 E4M3
workspace and selects TRT-LLM MHA decode. `TRTLLMBackend` uses XQA on SM120,
consuming the legacy linear block-scale layout with BF16 query/output. Its
native NVFP4 prefill guard requires SM100 GenMHA instead. The original claim that
all TRT-LLM MHA paths are SM100-only was therefore too broad. The upstream SM120
NVFP4-KV CI test covers Llama8B; that does not establish compatibility or quality
for this Qwen27B hybrid geometry. Its own queued probe is still required.

Evidence will land in `results/precision-reference-probes.json`. Its reported
capacity budget hint checks16,384 output tokens plus the263-token longest math
prompt; it is not a completed quality result. This queue can overlap resumed
host downloading, so its startup/request times are not clean performance
comparisons. No-fallback policy remains unverified.


## NVFP4 workspace and sizing follow-up — 2026-11-12

The current FlashInfer-prefill/TRT-LLM-MHA-decode recipe uses one cross-layer
FP8 K/V dequant workspace, not one pair per full-attention layer.
`NVFP4KVCacheMethod.create_buffers` creates that single pair; the pool's
`get_dequant_workspace` returns it to each layer. Its `compute_cell_size`
includes the shared workspace, and `PoolConfigurator._compute_cell_size`
uses that method after configuring the same resolved prefill/decode backends.

A real metadata-only constructor run at75,520 pool tokens plus64 padding rows,
4 KV heads,256 dimensions and16 full layers produces exactly1,547,960,320
buffer bytes, including147.625MiB of shared FP8 workspace. The cell cost is
18,432 persistent packed/scaled bytes plus2,048 workspace bytes per token,
20,480 total; same-KV-budget nominal ratio to FP8 is1.6. The prior1.78 estimate
excluded this workspace. Per-layer global FP32 scales add128 constant bytes.
This disproves persistent per-layer workspace duplication and an omitted
workspace in this cell estimator; it does not account for transient gathers,
BF16/FP8 dequant outputs, graphs, allocator reserves, state, weights or desktop.
No serving code changes or GPU execution are made. Current slowdown and
physical fallback/residency remain unresolved.

[Native constructor accounting and probe snapshot](../results/memory/kv-nvfp4-full-storage-accounting-20261112.json).
Sources: `layers/quantization/fp4_kv_cache_quant_method.py`,
`mem_cache/memory_pool.py`, `model_executor/pool_configurator.py` in the fixed
engine checkout.

### NVFP4 prefill versus decode access

The captured NVFP4-KV full-quality probe resolves FlashInfer prefill and
TRT-LLM MHA decode, with page64 and no speculation. On SM120, the inspected
TRT-LLM backend selects XQA. Its NVFP4 cache helper reads raw packed K/V and
linear block scales, which the decoder passes to the FlashInfer TRT-LLM decode
API. These inspected decoder functions do not call the FlashInfer dequant
workspace preparation routines. This source route does not support a separate
full-pool FP8 dequantization on every decode token as an explanation for the
current slowdown; it does not establish actual kernel timing or residency.

FlashInfer prefill instead gathers each request's actual cached prefix using
`req_to_token[req_idx, :prev_len]`, rather than gathering the entire allocated
pool. The native NVFP4 quant method dequantizes those selected rows to BF16 K/V
temporaries and converts them to FP8 for the shared workspace. These gathers
and BF16/FP8 outputs are additional transient payloads, distinct from the
persistent shared workspace already included in cell sizing. Their live peak
and cost remain unmeasured; a source call graph is not a GPU allocation trace.

[Configuration, source hashes and inspected function ranges/calls](../results/memory/kv-nvfp4-route-source-audit-20261112.json).
No serving source, active process or evaluation protocol changed for this audit.
