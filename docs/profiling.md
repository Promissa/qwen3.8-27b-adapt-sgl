# Measured baseline decode profile

The baseline spends most GPU kernel time in matrix projections. The priority is
FP4 GEMM backend selection and fitting an effective speculative draft; a new
sampling kernel has almost no available time to save in this trace.

Evidence: `results/profile/baseline-clean-analysis.json`, produced by
`bench/analyze_profile.py` from the clean 32-step decode trace under
`results/validation/baseline-near-limit-088/profile/`. The original failed cleanup
attempt also produced a valid independent trace; its analysis is retained as
`results/profile/baseline-analysis.json`.

The source is SGLang c3c292b65a plus the isolated Qwen3.5 text-only CLI fix. Target
revision is 5b7a687fc8211a5d631c8ca6a593dd37eb26ce33, with FP8 KV, BF16 recurrent
state, FlashInfer attention and the default SM120 CUTLASS FP4 path, concurrency 1.

| Component | Share of summed GPU kernel duration | Summed time per decode step |
|---|---:|---:|
| MLP GEMMs | 56.10% | 8.30 ms |
| GDN QKVZ/output projection GEMMs | 19.52% | 2.89 ms |
| GDN BF16 BA projection GEMV | 4.64% | 0.69 ms |
| Full-attention projection GEMMs | 5.91% | 0.87 ms |
| LM head GEMM | 4.37% | 0.65 ms |
| FP4 activation quantization | 3.01% | 0.45 ms |
| GDN recurrence, convolution and output norm | 2.20% | 0.33 ms |
| Full-attention core, RoPE and KV storage | 1.30% | 0.19 ms |
| Residual normalization | 1.67% | 0.25 ms |
| Sampling | about 0.01% | about 0.002 ms |
| Other pointwise/bookkeeping kernels | 1.27% | 0.19 ms |

These are **overlapping kernel durations, not wall-time percentages**. Across 32
steps, their sum is 473.49 ms, union is 444.97 ms, and GPU timeline span is
463.82 ms. Profiling changes execution timing. The independent serving benchmark
is authoritative for throughput: three repeats each give mean 69.449 tok/s at
1024/1024 and 65.389 tok/s at 8192/1024. Median TPOT across repeats is 14.333 ms
and 14.694 ms respectively.

Attribution checks all 32 × 64 layer GEMM sequences against the source geometry:
three GDN layers followed by one full-attention layer, two attention/GDN
projections plus two MLP projections per layer, then one LM-head projection.
There are 257 CUTLASS GEMMs per decode step. Kernel grid/order mismatches abort
analysis instead of assigning an assumed category. This mapping is specific to
this baseline and cannot be applied blindly to cuDNN or speculative traces.

Host annotations and CUDA synchronization are reported separately in the JSON.
For example, `scheduler.process_batch_result` includes waiting for GPU output;
its inclusive duration is not additional CPU compute overhead. The trace shows
overlap across streams, so subtracting summed GPU durations from host durations
would be invalid.

The allocator snapshot records actual reserved, allocated and active bytes at
capture time. This omits Windows desktop memory and CUDA allocations outside
PyTorch. Device-wide sampled peaks and load/cache/graph allocation lines remain
in each probe's raw evidence; the two measurements must not be substituted for
one another.

The original relative targets remain: at least 2× context and 1.5× single-stream
throughput at comparable desktop occupancy, with the stated quality gates. Against
the measured speed control, 1.5× means about 104.17 tok/s for 1024/1024 and
98.08 tok/s for 8192/1024. The two traces support testing GEMM and speculative
paths first. They do not establish that either target has already been achieved.

## Resumed-session runtime observation (2026-11-11)

The new vision.92/pin4/24576 probe records3314MiB already used (initial host
snapshot3299MiB), versus
1245MiB for the interrupted r1 cohort. Its capacity8192/1024 request takes47.54s
(vs roughly15.9s before); this single capacity request is not a formal repeated
benchmark or a controlled speed comparison.

Three read-only whole-device hardware windows during the unchanged mixed soak
show P0, SM2880–2932MHz, memory14001MHz and49–51°C. Across six PCIe readings,
RX is6119–23454MB/s and TX7270–20589MB/s. The three device free-memory
samples are561–568MiB. These sampled high clocks do not show an obvious
frequency collapse at those moments, while PCIe traffic is large. They do not
attribute traffic to this process, prove paging/shared-memory fallback, establish
residency, or certify Windows policy. No configuration changed and the observer
sent no server requests.

Raw commands/output and the verified current owner are retained in
`results/vision-runtime-r2-hardware-observation.json`; independently parsed
ranges and the raw SHA256 are in
`results/vision-runtime-r2-hardware-summary.json`. The baseline profile and
69.45/65.39tok/s control above belong to the earlier cohort; keep them separate.

## 2026-11-19 单流性能复盘与下一轮实验

在 RTX 5090 D v2 上，**长输出的首要问题是一次目标模型前向能提交几个正确 token，长文档问答的首要问题是第一次读完输入要多久**。两者在真实应用上占用的时间不同。以下采用同一轮的原始请求重新汇总。更早的容量扫描、数值定位、精度对照、前后基线漂移和失败草稿用于解释机制。[实验台账](../results/ledger.md) · [容量扫描](../results/envelope-present-validated/table.md) · [11 月 12 日缩减测试](three-hour-delivery.md) · [11 月 19 日目标](../results/continuation-20261119/spec/runs.json)

### 实测差距

原生流式接口，单并发，每个长度三次，逐次清缓存，固定生成 1,024 token。表中为请求中位数。解码速度按首末 token 间隔计算，端到端用客户端发起到结束的时间。

| 配方 | 1K / 8K 解码速度，tok/s | 1K / 8K 冷 TTFT，ms | 1K / 8K 端到端，s |
|---|---:|---:|---:|
| 原始权重参考，49,152 池 | 64.39 / 63.54 | 87.7 / 667.7 | 15.98 / 16.77 |
| Balanced，49,152 池 | 63.08 / 62.56 | 86.7 / 687.9 | 16.31 / 17.04 |
| Max-context，81,920 池 | 64.97 / 63.42 | 82.6 / 631.7 | 15.90 / 16.78 |

按当前目标，1K 冷启动后的首 token 已低于 500 ms。8K 的目标为 550 ms，Balanced 需缩短 138 ms（20%），Max-context 需缩短 82 ms（13%）。这里的“冷”指已就绪的服务清空前缀缓存。原始的 1.5× 单流指标须与同协议、同时间段的对照比较，另报 90–100 tok/s 绝对目标。Cookbook OAI 接口三次 8K 端到端输出吞吐中位数为 Balanced 57.68、Max-context 62.91 tok/s，包含首 token 和流式传输。[参考请求](../results/continuation-20261119/application/reference/serving.json) · [Balanced 请求](../results/continuation-20261119/application/balanced/serving.json) · [Max-context 请求](../results/continuation-20261119/application/max-context/serving.json) · [Cookbook Balanced](../results/bench/continuation-20261119-balanced) · [Cookbook Max-context](../results/bench/continuation-20261119-max-context)

### 单 token 解码与推测验证

- **主干的权重投影占用最多 GPU 时间。** 该模型 64 层中有 48 层 GDN 和 16 层全注意力。旧基线的 32 步 trace 验证了每步 257 个 CUTLASS GEMM。MLP 占 GPU kernel 时长之和的 56.1%，GDN 的 QKVZ 和输出投影占 19.5%，另有 BF16 BA 投影 4.6%。GDN 递推、卷积和输出归一化合计 2.2%，采样约 0.01%。这些比例含跨流重叠，当前驱动和生产 CUDA Graph 下还需复测。16K 固定池的 A/B/B/A 显示，量化输入 embedding 节省 1.64 GiB，候选解码速度仅为对照的 0.990× / 0.985×。单 token 输入只查少量 embedding 行，主要投影依然每步执行。[算子 trace](../results/profile/baseline-clean-analysis.json) · [权重格式匹配对照](../results/continuation-20261119/matched-serving/comparison.json)
- **90–100 tok/s 对纯逐 token 路径非常苛刻。** 从实际 checkpoint 的分组字节数扣除视觉塔和输入 embedding，余下约 14.45 GB。若每步从显存读取一次这些权重，按标称 1,344 GB/s 计算，90 tok/s 就需约 1,301 GB/s，理想速率约 93 tok/s。这是模型化估算，运行时布局、缓存、KV 和状态流量都会改变结果。它解释了为什么值得优先尝试让多个候选 token 共用一次权重读取。是否已经打满显存带宽，还需看实际 DRAM 读量和耗时。[张量清点](../results/target-tensor-audit.json) · [NVIDIA 显卡规格](https://www.nvidia.cn/geforce/graphics-cards/50-series/rtx-5090-d-v2/)
- **目前达到逐 token 一致的 NGRAM，接受长度太短。** 原权重、确定性 Triton、4K 池、eager、开 logprob 的 8×512 请求全部一致。3,080 次验证提出 9,240 个草稿 token，接受 1,010 个（10.9%），其中 2,380 次（77.3%）一个草稿也没接受。每次含保底 token 平均提交约 1.33 个。端到端时间合计 303.26→265.29 s，得到 1.143×，其中两条请求更慢。假设每轮验证与普通解码一步等时，当前接受分布也只支持约 1.33× 的解码收益。把观测端到端比例作为粗略代理，在轮耗时比不变时，达到 1.5× 约需平均 1.74 个提交 token。后一个数字包括 prefill、logprob、主机开销的影响，需要单独计时验证。[逐题接受直方图与计时](../results/continuation-20261119/spec/latency-comparison.json)
- **高接受率草稿受数值一致性限制。** 原生 EAGLE 的 8 个请求曾报告平均接受长度 2.82、解码速度中位 125 tok/s，但 0/8 与其 greedy 对照相同。DFlash 的确定性实验接受长度 3.86，同样 0/8。DSpark 在一个 Triton 组合只有 8,297 个池 token，另一个可执行组合仍有 token 分歧。GDN 的 BF16 状态写回、树卷积累加顺序以及全注意力 verify 都曾定位到路径差异。单独改变 attention 的 KV tile 能修复保存的算例，整模依然 0/8。11 月 19 日的 8/8 来自受限的 eager decode-attention 路径。这条路径在构造可见 KV 索引时读回标量、按掩码分配张量，并主动拒绝图捕获。各项开销占比还缺实测。[草稿实测](gdn-verify-investigation.md) · [注意力回放](attention-verify-findings.md) · [验证分支](../engine/sglang/python/sglang/srt/layers/attention/triton_backend.py)

### 冷请求、命中前缀和应用耗时

- **8K 冷 TTFT 的差距在服务端前向阶段。** Balanced 三次请求的调度时间戳显示，从首次进入前向到 prefill 结束分别耗时 656、682、692 ms，排队均低于 0.7 ms。服务日志显示四个 2,048-token chunk、缓存命中 0。该阶段包括分块调度和 GPU 工作，尚无当前 8K prefill 的算子级耗时。即使把 Balanced 冷 TTFT 降到 550 ms，固定输出 1,024 token 的 17.04 s 总时间也只缩短约 0.8%。[请求时间戳](../results/continuation-20261119/application/balanced/serving.json) · [服务日志](../results/continuation-20261119/application/balanced/probe/server.log)
- **热请求的 50 ms 目标需要优化命中后的路径。** Balanced 三次都是 48,680 输入，其中 48,640 命中、40 个新 token。客户端首段非空文本耗时 53.8 / 85.8 / 117.8 ms；服务端首 token 为 45.0 / 78.2 / 111.7 ms，其中进入前向到 prefill 结束为 36.1 / 70.3 / 102.6 ms。三次均满足原计划的 1.5 s，均高于收紧后的客户端 50 ms。样本不足以归因于检索深度。Max-context 采用单状态槽并关闭 radix，后续 81,440 token 整段重算，三次约 14.7 s。两套配方的长度和用途不同。[Balanced 检索](../results/continuation-20261119/application/balanced/retrieval.json) · [Max-context 检索](../results/continuation-20261119/application/max-context/retrieval.json)
- **应用耗时取决于任务形态。** 31 个共同可评分的 BFCL 会话，按每次模型调用的客户端耗时相加，参考 / Balanced / Max-context 为 513.8 / 524.0 / 704.9 s，分别调用 406 / 419 / 416 次，通过 13 / 18 / 16 题。参考另有一题超容量。工具轨迹和输出长度会随模型回答改变，计时也未含外部工具实际执行。16 道 LongBench 的请求耗时分别为 61.93 / 57.97 / 53.63 s，通过 6 / 7 / 6 题。Balanced 的 57.97 s 中有 45.58 s（78.6%）在首次前向到 prefill 结束之间，16 题缓存命中均为零。即使消除这个区间以外的全部时间，按这一批请求估算也只有约 1.27×。回答长度和封顶题数不同，三个配方的总时差不能直接归功于引擎。[应用原始记录](../results/continuation-20261119/application) · [配对质量和容量拒绝](../results/continuation-20261119/application/paired-quality.json)

### 建议实验顺序与晋级条件

| 优先级 | 实际改动与对照 | 观测值与决策 |
|---|---|---|
| 1 · Balanced 冷 prefill | 保持 49,152 池、4 状态槽、C1、FP8 KV、BF16 状态。从当前 2,048 对比 4,096 chunk，记录实际 graph 捕获形状和每块 GPU 时间。在固定 8K 输入标记 MLP、48 层 GDN 的投影和扫描、16 层全注意力，以及 chunk 间主机间隔。仅当较大块确实节省时间且保留容量时再试 8,192。 | 首要目标是 8K 冷 TTFT 中位 ≤550 ms；同时验证 48,640 真实检索。若开销来自 GPU 内核，针对 prefill 中耗时最高的形状做后端或 tile 对照；若来自 chunk 边界，才改调度或 graph 覆盖。用 4 道日常 LongBench 检查是否传导到长文档请求。 |
| 2 · Balanced 热前缀 | 复用实测的 48,640+40 模板，固定后缀，改变深度顺序并重复。给 UnifiedRadixCache 匹配、GDN checkpoint 取用/复制、40-token 有前缀 extend、首 token 传输分别打点。对全注意力记录实际历史 KV 读取。 | 逐次同时报告命中 token、客户端首文本和服务端首 token。若长 KV 的短 extend 占主导，做小 query、长 prefix 的 attention 分块对照；若时间花在状态复制/分配，才试同请求生命周期内复用目标缓冲区，覆盖逐出、重启和状态隔离。客户端检验 ≤50 ms，保留三深度检索。 |
| 3 · 无损长输出 | 用已通过的 C1 eager NGRAM 做数值参考。在固定 draft 宽度下，把可见 KV 索引构造和 split 计算移到设备端，按上限预分配、逐批刷新索引与 scratch，保留树掩码与有效长度。先覆盖连续序列、分叉、拒绝回滚、复用旧槽，再验证 CUDA Graph capture/replay。随后以已能容纳的 early-share Qwen MTP/EAGLE 为首选，逐步移植验证路径；按首次分歧核对 logits、GDN 已提交状态和祖先 KV。 | 每个改动先用已知失败题和留出题定位，再验证固定 8×512 token 完全一致。报告真实 graph replay、草稿/验证/提交时间、接受直方图、每轮提交数和目标模型前向次数。若 MTP 仍有分歧，保留在实验分支。NGRAM 低重复请求可试按实测单位提交成本退回普通 decode，另测切换处的状态和 token 一致性。 |
| 4 · 普通解码内核 | 在当前生产 graph 上复采 32 步，优先隔离 M=1 的 MLP gate/up、down 和 GDN QKVZ/out 的真实权重形状、打包格式和 scale。用 DRAM 读量、L2 命中率、SM 活跃度和 GPU 时间决定是否试小 M 的 SM120 GEMM tile 或消除中间量读写。先核对仓内已有 SiLU×FP4 融合的实际 dispatch。 | 单内核改善须传导到 1K/8K 的 TPOT。若已接近实测的可持续读带宽，集中做多 token 验证。采样器、skip-decode-lock、graph pool 预留和更低精度 KV 排在后面；已有试验显示收益很小或伴随质量和长前缀代价。 |

为缩短总评测时间，每项先跑固定输入的配对短测。11 月 19 日更新后，最终评测前原定四轮及以上的重复测速缩至两轮，保留逐轮值并取算术平均，记录桌面占用、池大小、权重哈希、图配置和逐次结果。正确性抓取可开 logprob，正式测速双方均关闭 logprob，分别报严格同模式对照与当前生产 CUDA Graph 基线上的绝对速度。无损 1.5× 还须在相同输出 token 上比较总耗时。BFCL 固定工具回放用于隔离引擎收益，冻结清单的自由工具轨迹用于测实际应用耗时，连同调用数、通过数、拒绝和截断一并报告。先跑 8+4 日常子集，只对晋级配方补做 24+12 留出。历史同一对照漂移约 18%，单次峰值不用于晋级。[评测计划](../PLAN.md) · [历史漂移](three-hour-delivery.md) · [Nsight Compute 计数器说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## 11 月 19 日加速筛查实测

全部 GPU 实验在单卡串行运行。取消稳定性测试后，保留 Balanced 的 1,169 秒部分记录，Max-context 的质量复测独立运行。每种重复测速保留两轮并取算术平均。8 个严格提示、3 个检索深度、4 道 LongBench 和 8 个 BFCL 会话均保留各自的不同输入。[原始试验目录](../results/fast-20261119) · [复算汇总](../results/fast-20261119/comparison.json)

### 冷 prefill 与长前缀复用

同一份 Balanced 权重、49,152-token 池、4 个状态槽。冷测先预热形状，每次清缓存，用固定随机种子的 8K 输入生成 4 个 token 以测首 token。热测使用 48,640 个缓存 token 加 40 个后缀 token，两轮的检索深度顺序相反。

| 配置 | 8K 冷 TTFT 两轮均值 | 热 TTFT 两轮均值，10% / 50% / 90% 深度 | 检索 | LongBench 四题通过数 / 请求耗时总和 |
|---|---:|---:|---:|---:|
| chunk 2,048 | 652.2 ms | 70.8 / 68.5 / 83.0 ms | 6/6 | 3/4，14.14 s |
| chunk 4,096 | 704.9 ms | 58.9 / 50.7 / 56.6 ms | 6/6 | 3/4，14.96 s |
| chunk 2,048，启用 GDN prefill 图捕获，图内并发上限 1 | 641.3 ms | 47.0 / 48.2 / 49.8 ms | 6/6 | 3/4，13.49 s |

GDN 候选的 6 个热请求有 4 个单次达到 50 ms。相同 2,048 分块和 40-token 后缀的独立 trace 中，48 层 GDN 纳入图捕获后，CUDA Graph 启动数从 65 减到 17，GPU kernel 累计耗时为 24.70 / 26.89 ms。这支持减少主机启动开销的解释。冷请求仍超过 550 ms，4,096 分块未晋级。可用以下命令复现热前缀候选。[对照](../results/fast-20261119/chunk-2048/screen/screen.json) · [4,096 分块](../results/fast-20261119/chunk-4096/screen/screen.json) · [GDN 图捕获](../results/fast-20261119/gdn-prefill-graph/screen/screen.json)

```bash
SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND=1 SGLANG_GDN_PREFILL_GRAPH_MAX_SEQS=1 bash scripts/serve-balanced.sh
```

GDN 图捕获的 BFCL 筛查为 4/8，与此前 Balanced 同分。逐题有一得一失，LongBench 四题的判分一致。对新失分的 `multi_turn_long_context_121` 单独复测，本轮普通 Balanced 与 GDN 候选均失败，无法将其稳定归因于图捕获。另补完原队列中 Max-context 的质量分歧复测，BFCL 0/3、LongBench 0/2；原始权重参考在这些题复测中分别为 1/3 和 2/2。这些是定向复测的分母。[GDN 筛查](../results/fast-20261119/gdn-prefill-graph/bfcl/summary.json) · [对照复测](../results/fast-20261119/balanced-control/bfcl-recheck/summary.json) · [GDN 复测](../results/fast-20261119/gdn-bfcl-recheck/bfcl/summary.json) · [Max-context 复测](../results/fast-20261119/max-quality-recheck)

### 图内逐 token 验证与 MTP

在原始 NVFP4 目标权重、固定 16,384 池、确定性 Triton、C1/D4 上，为每行查询在 GPU 上构造可见前缀和草稿祖先的 KV 索引，将指针、分块数和临时输出放入可复用的工作区，再调用原生 decode attention。CUDA Graph 在实机捕获并回放。隔离回放覆盖空前缀、长短变化、树分叉和槽位复用。首轮捕获曾因 upstream 的 uint8 图掩码被实验分支拒绝，加入该布局并以真实类型回放后完成整模比较。[源码补丁](../patches/0007-qwen-c1-verify-decode-graph.patch) · [设备回放](../results/fast-20261119/graph-indices-u8.json)

| 配方 | 固定 8×512 greedy | 两轮定长请求平均时间 | 相对同模式对照的端到端加速 | 定长请求每轮验证平均提交 token |
|---|---:|---:|---:|---:|
| 关闭推测的 CUDA Graph 对照 | 4,096 个 token 与原 eager 对照相同 | 10.16 s | 1.000× | 1 |
| NGRAM + 图内 verify | 8/8 相同 | 9.53 s | 1.067× | 1.25 |
| 外置量化 Qwen MTP + 图内 verify | 8/8 相同 | 4.22 s | 2.407× | 2.80 |

每个计时轮包含两个不同的真实提示，各固定生成 256 token，关闭 logprob，四个计时响应也逐 token 比较。另将 8 条 512-token 请求各执行一次，MTP 总耗时 33.37 s，对照 76.42 s。两轮筛查仅覆盖本表的模型、精度、缓存和请求设置。复现 MTP 用 `bash scripts/serve-mtp-graph.sh`，启动脚本的 26 个有效参数与 MTP 成功试验核对一致。[MTP 启动脚本](../scripts/serve-mtp-graph.sh) · [对照记录](../results/fast-20261119/graph-r2/control/result.json) · [NGRAM](../results/fast-20261119/graph-r3/ngram/result.json) · [MTP](../results/fast-20261119/graph-r3/mtp/result.json)

在独立重启的相同 16K 控制/候选上，原生流式接口使用相同随机输入 ID、清缓存、单并发、关闭 logprob，固定生成 1,024 token。每长度两轮取算术平均，按四个 SSE 文件末帧的完整输出 ID 与对照比较，4/4 均逐 token 相同。两组都使用完整目标 decode/verify CUDA Graph。四个合成请求的 MTP 接受长度为 3.89–3.95，高于真实提示定长请求的 2.80，两个工作负载分别报告。

| 合成输入 | 对照 → MTP 解码速度，tok/s | 对照 → MTP 请求耗时 | 端到端加速 | 对照 → MTP 冷 TTFT |
|---|---:|---:|---:|---:|
| 1,024 token | 55.23 → 172.95 | 18.65 → 6.09 s | 3.06× | 129.8 → 177.8 ms |
| 8,192 token | 54.72 → 167.65 | 19.41 → 6.84 s | 2.84× | 711.8 → 738.9 ms |

在这组前置筛查中，MTP 达到 90–100 tok/s 和 1.5× 的目标，8K 冷 TTFT 仍高于 550 ms。GDN 热前缀候选使用不同权重及 49K 池，其数据独立于 MTP。这一轮尚未运行二者组合或 MTP 的最终应用留出集。[流式原始对照](../results/fast-20261119/native-graph/control/metrics.json) · [流式 MTP 原始记录](../results/fast-20261119/native-graph/mtp/metrics.json) · [跨运行逐 token 核对](../results/fast-20261119/comparison.json)

### 普通解码

当前生产 Balanced 的 32 步 trace 中，每步仍为 257 个 CUTLASS GEMM，MLP 投影占累计 kernel 时长的 53.9%，GDN QKVZ/输出投影占 19.1%。按当前实际 grid 检查了全部 32×64 层的投影顺序。将原有 SiLU+FP4 输入量化融合额外用于 16 个全注意力层后，8 个严格样本通过，真实请求端到端为 0.999×。两轮固定 256 输出的 1K / 8K 解码速度分别为对照 62.39 / 61.38、候选 61.74 / 60.86 tok/s。保留默认关闭。[当前 trace](../results/fast-20261119/balanced-control/decode-analysis.json) · [扩展融合试验](../results/fast-20261119/full-mlp-fused) · [可选补丁](../patches/0008-qwen-full-attention-mlp-fusion.patch)

## 实测上下文极限与 80% 输入

RTX 5090 D v2 24 GB、Ryzen 7 9700X，WSL2 下单卡单并发。对每个部署配方解除手工指定的总 token 池上限，保留显存比例、状态槽、模型和后端。读取调度器边界，再发送一对相邻长度请求：较短者完整读入并生成 1 token，较长者因长度返回 HTTP 400。测速使用最大成功输入的 80% 向下取整，实际生成 1,024 token。每个长度清缓存、相同种子两轮，记录算术平均；端到端加速比等于对照的平均总耗时除以候选的平均总耗时。该容量适用于这些配方和启动时的桌面占用，Windows 的显存回退状态仍未获独立确认。[实验计划](../PLAN.md) · [原始数据](../results/context80-20261119)

### MTP 与同池对照

原始 NVFP4 权重和确定性 Triton 后端，4 个状态槽，静态显存比例 0.92。目标模型和外置量化 MTP 可使用 57,452-token 池，两个启动均实测到 57,445 个输入 token，80% 点为 45,956。关闭 MTP 的对照以相同池、相同最大长度预热，才开始计时。以下解码率仅计首个输出到最后一个输出之间的区段；长输入端到端速度包含完整 prefill。

| 输入 token | 对照 → MTP 解码 tok/s | 对照 → MTP 冷 TTFT | 对照 → MTP 端到端 | 加速比 |
|---:|---:|---:|---:|---:|
| 1,024 | 55.48 → 49.68 | 0.127 → 0.318 s | 18.57 → 20.91 s | 0.888× |
| 8,192 | 54.97 → 52.08 | 0.689 → 1.909 s | 19.30 → 21.55 s | 0.895× |
| 45,956 | 50.67 → 38.32 | 6.476 → 21.241 s | 26.67 → 48.28 s | 0.552× |

六次定长请求的完整 1,024-token 输出全部与对照一致。45,956 点的三个检索深度在首问和续问中均正确。16 道冻结 LongBench 均逐 token 一致，两边都答对 6 道，有 3 个回答达到 128-token 输出预算。80% 长度附近有 3 道符合 ±5% 窗口的原始长文题，两边都为 0/3，每题均返回可解析的选项，未触及输出上限。其中 2 道与冻结集合重叠。BFCL 均为 3/8，全部 8 个会话的工具调用与回复结构完全一致，各 80 个生成步骤，均未触及 512-token 上限。对照和 MTP 的 BFCL 推理用时之和分别为 134.8、238.2 秒。三处精确检索通过说明模型能取回显式标记；长文选择题仍暴露理解短板。[逐项核对](../results/context80-20261119/mtp-comparison.json) · [近 80% 原题](../results/context80-20261119/mtp/natural80-ids.json)

该配方在 57,452 池下出现反向加速。尽管 1K 请求平均每次验证仍提交约 3.9 个 token，设备峰值占用达到 23,712 MiB，对照为 22,639 MiB。两次启动前的设备占用分别是 2,518 与 2,044 MiB。运行期间 MTP 的三次 PCIe 采样分别观察到 8.3–13.7 GB/s 的发出流量，同期 Windows 报告最繁忙适配器的共享内存约 517 MiB。结果与显存压力和额外传输相符，适配器级计数无法判断哪些页属于推理进程。此前 16K 池的 2.84×（8K 输入）和这里的 0.895× 使用不同的缓存预算。[MTP PCIe 采样](../results/context80-20261119/mtp/pcie-sample.txt) · [Windows 只读计数](../results/context80-20261119/mtp/windows-adapter-sample.json)

同一 45,956-token 工作负载另做了一组限池配对。48,128-token 池给固定输出和未裁剪长文题都留下生成空间。启动时对照和 MTP 的设备占用仅相差 3 MiB（1,157 / 1,160 MiB）。关闭 / 启用 MTP，解码为 **50.42 / 134.43 tok/s**，冷 TTFT 为 **6.54 / 6.91 s**，端到端为 **26.83 / 14.54 s（1.845×）**。两次 1,024-token 输出均逐 token 相同，两边三道自然长文题的输出也逐 token 相同，均为 0/3，检索及续问均为 6/6。这组适合长输入、长输出的测点验证了 1.5× 目标。和上面 57,452 池比较时，池容量与桌面占用同时变化，无法将全部收益归给缩池。复现这组启动命令可用 `bash scripts/serve-mtp-graph.sh --max-total-tokens 48128`。[同池配对数据](../results/context80-20261119/fit-comparison.json)

按相同三道长文题的起止时间对齐整卡 PCIe 采样，57,452 池 MTP 的发送流量中位数为 10,216.5 MB/s（14 个样本），同池对照为 57.0 MB/s（4 个样本）。48,128 池的 MTP 与对照分别为 39.5、37.5 MB/s（各 4 样本）。这给出了延迟变化时的传输量实测，进程级迁移页数仍然未知。[分窗口采样及原值](../results/context80-20261119/pcie-longbench.json)

限池 MTP 的三道自然长文题每题只生成 8 token。对照合计 19.62 秒，MTP 合计 20.83 秒（0.942×）。对长文分类、短答等输出很短的请求，冷 prefill 足以抵消解码收益；1.845× 对应固定生成 1,024 token 的负载。

### GDN 长前缀图捕获

使用转换后的 NVFP4 embedding、FP8 KV、4 个状态槽、2,048-token 分块。默认静态显存比例 0.88 时，自动池为 149,937。149,931 输入被 SGLang 拒绝，149,930 输入进入 prefill，但在 180 秒客户端时限内没有生成 token。将比例调到 0.815，候选自动池和对照固定池同为 102,835。两个服务均成功完整处理 102,828 输入并生成 1 token，也均拒绝 102,829。新增长度是 82,262，另用 MTP 的 45,956-token 输入做同形状交叉测量。GDN 开、关使用相同权重和输入；下表读出完整冷请求。GPU 日志确认候选把 GDN extend 放进 prefill 图。[默认比例探测](../results/context80-20261119/gdn/probe/validation-00.log) · [成功边界](../results/context80-20261119/fraction-gdn-815/gdn/boundary.json) · [开关读回](../results/context80-20261119/fraction-gdn-815/gdn/live-environment.json)

| 输入 token | 对照 → GDN 捕获解码 tok/s | 对照 → 捕获冷 TTFT | 对照 → 捕获端到端 | 加速比 |
|---:|---:|---:|---:|---:|
| 1,024 | 65.16 → 62.86 | 0.111 → 0.117 s | 15.81 → 16.39 s | 0.965× |
| 8,192 | 63.35 → 61.15 | 0.652 → 0.660 s | 16.80 → 17.39 s | 0.966× |
| 45,956 | 59.99 → 57.50 | 5.989 → 6.160 s | 23.04 → 23.95 s | 0.962× |
| 82,262 | 56.43 → 54.36 | 15.072 → 15.465 s | 33.20 → 34.29 s | 0.968× |

82,262-token 检索的续问每次实际命中 82,240 token，另有 62 token 要计算。三个深度各两轮，开关两边均答对 12/12 个首问和续问。图捕获将热 TTFT 总均值由 64.82 降到 56.84 ms（减少 12.3%），三个深度的捕获均值依次为 59.36、54.13、57.02 ms。此前 48,640 命中、40 新 token 的筛查达到 50 ms，本组在更长缓存和 62-token 后缀下未达该值。四种长度的 8 个定长输出都与同池对照逐 token 相同。[对照检索](../results/context80-20261119/fraction-gdn-815/gdn-control/retrieval80.json) · [GDN 检索](../results/context80-20261119/fraction-gdn-815/gdn/retrieval80.json)

GDN 开关两边的固定 LongBench 均为 6/16，有 5 个回答达到 128-token 预算；16 题逐 token 相同。80% 附近原题长 80,199–82,325 token，覆盖单文档问答和代码仓库理解；两边均为 1/4，4 题逐 token 相同，2 题达到输出上限且没有可解析选项。BFCL 均通过 4/8，8 个会话的完整工具调用及回复一致，各 86 个生成步骤，其中 1 步触及 512-token 上限。GDN 配方和 MTP 配方使用的目标权重不同，开关影响按各自对照判定。[GDN 逐项复算](../results/context80-20261119/gdn-comparison.json)

### Max-context 容量与长输入代价

关闭 radix，使用转换 embedding 和 1 个状态槽。静态显存比例 0.92 时，池大小为 186,104；186,098 输入被服务拒绝，186,097 输入在 180 秒内未返回首个 token。改用 0.8775 后，池大小为 155,307，**155,300 个输入在 45.36 秒内完整读入并生成 1 token，155,301 被拒绝**。本轮已完成生成的最大输入是 155,300，相应 80% 为 **124,240**。原始 180 秒超时记录没有给出更高长度的成功值。124,240 超过此轮两种加速配方各自验证的输入边界，因此共同长度对照使用 45,956。[较大池的超时](../results/context80-20261119/max/boundary.json) · [相邻长度实测](../results/context80-20261119/fraction-max-8775/max/boundary.json)

| 输入 token | 冷 TTFT | 解码 tok/s | 端到端，含固定 1,024 输出 |
|---:|---:|---:|---:|
| 1,024 | 0.117 s | 65.37 | 15.77 s |
| 8,192 | 0.653 s | 58.24 | 18.42 s |
| 124,240 | 30.459 s | 53.45 | 49.60 s |

124,240 输入处，三种深度的首问、续问全部取回正确密钥（6/6）。关闭 radix 后，续问有 124,281 个输入 token、缓存命中为 0，平均首 token 30.64 秒。另取 120,191–124,161 token 的四道未裁剪 LongBench 原题，两道文档、两道代码仓库题，得分 **0/4**。两题达到 128-token 预算仍未产生可解析选项，其余两题给出错误选项。此组只评最大容量配方的绝对表现。短密钥检索与整篇文档理解在同样长度下有明显差距。[检索原始输出](../results/context80-20261119/fraction-max-8775/max/retrieval80.json) · [长文逐题答案](../results/context80-20261119/fraction-max-8775/max/near80-longbench.json)

### 复现与范围

从项目根目录执行下列命令可复跑单卡矩阵，新运行应使用新的结果目录。脚本先核对边界并填充到相同高水位，再进行控制和候选比较。同一结果目录重跑会复用有退出记录的试验并重新核对数据。两组应用评测均使用本地冻结的数据、模板和官方评分流程；只选取按长度预先确定的近 80% 题，未补长、裁剪或按结果筛题。固定 16 题和近 80% 题分别计分，BFCL 本次采用 8 个完整会话。未测试两项加速的叠加。小样本结论限定在这些量化权重、greedy 设置和输出预算。[完整复算数据](../results/context80-20261119/comparison.json) · [协议及权重版本](../results/context80-20261119/protocol.json)

```bash
source scripts/runtime-env.sh
.venv/bin/python scripts/run_context80.py --root results/context80-rerun
```

### 65,536-token 同池 MTP 配对（2026-11-20）

单卡、单并发，使用转换后的 NVFP4 embedding，两个服务均使用 FP8 KV、BF16 GDN 状态、确定性 Triton、4 个状态槽、1,024-token prefill chunk 和相同的 GDN/MLP 开关。基于相同的两组随机种子生成 64,000-token 输入，逐请求清缓存，固定贪心输出 1,024。两轮算术平均如下。

| 指标 | 无 MTP | C1/D4 MTP |
|---|---:|---:|
| 冷 TTFT | 11.090 s | 11.680 s |
| Decode | 46.84 tok/s | 123.80 tok/s |
| 端到端 | 32.931 s | 19.944 s |

端到端 1.651×，两条输出均逐 token 相同，两侧零回退。MTP 平均每轮验证提交 3.95 token。启动前整卡占用为 2,736 / 2,453 MiB。这个合成、长输出测点把已验证输入下界从 45,956 提升到 64,000。原 48,128 池实验使用另一份目标 embedding，跨实验差异无法独立归因。当前两条输出的相等性仅覆盖这两个输入。[运行脚本](../results/kv-window-20261120/screen.sh) · [配对复算](../results/kv-window-20261120/comparison.json)

### 81,920-token 池的短输出筛查（2026-11-20）

Qwen3.8-27B 在 RTX 5090 D V2 24 GB 上使用同一 NVFP4 embedding、FP8 KV、4 个 GDN 状态槽和 1,024-token prefill chunk。关闭与开启 C1/D4 MTP 各跑同样两条 80,000-token 冷输入，每条固定贪心生成 256 token。两侧均零缓存命中、零回退，两条输出逐 token 一致。两侧启动前整卡显存分别为 3,912 和 4,022 MiB。

| 两轮平均 | 无 MTP | C1/D4 MTP |
|---|---:|---:|
| 冷 TTFT | 15.87 s | 16.19 s |
| 解码 | 44.10 tok/s | 113.50 tok/s |
| 端到端 | 21.66 s | 18.44 s |

80,000 输入、256 输出的端到端加速为 **1.174×**，达到 1.00× 门槛。三条真实应用样本均提前结束，输入分别是 79,626-token 长文问答、49,578-token CSV 字段抽取、64,786-token BLIP 代码定位。两侧均答对 3/3，输出 3/3 逐 token 一致。三条样本仅生成 2–5 token，MTP 侧单次端到端各多用约 0.55–1.07 秒。依据短测停止条件，本轮共执行 4 个测速请求和 6 个质量请求。[速度对照](../results/kv-window-short-20261120/speed-80000-slots4-comparison.json) · [质量对照](../results/kv-window-short-20261120/quality-80000-slots4-comparison.json)

### 131,072-token 池的 128K 短输出筛查（2026-11-20）

单并发，冷输入 128,000 token，每条固定贪心输出 256 token。原配方使用 4 个 GDN 状态槽及 prefill 图。MTP 侧成功分配 131,072-token FP8 KV 池，验证图捕获后报告空余 0.00 GiB。首轮端到端 52.51 秒，第二轮在两轮合计 180 秒的测速上限内没有完成。第二轮整卡 PCIe 发送速率约为 12.1 GB/s 中位数，最高记录 15.0 GB/s。这组计数无法归属到单个进程。[原配方诊断](/home/pr/projects/qwen3.8-27b-adapt-sgl/results/kv-window-short-20261120/speed-128000-slots4-eagle-native/probe/probe.json)

复测两侧均关闭 radix 和 prefill CUDA Graph，使用 1 个 GDN 状态槽。MTP 保留目标验证图和草稿解码图，捕获完成时报告 0.58 GiB 空余。常驻 SSM 状态由约 0.35 GiB 降至 0.14 GiB，验证快照仍为 0.56 GiB。两侧在同一个池容量下均完成两轮，零缓存命中，零回退，两条 256-token 输出逐 token 一致。MTP 侧先运行，启动前整卡占用约 1,669 MiB，无 MTP 侧为 2,016 MiB。原配方启动前占用约 2,779 MiB。

| 两轮平均 | 无 MTP | C1/D4 MTP |
|---|---:|---:|
| 冷 TTFT | 34.84 s | 40.19 s |
| 解码 | 43.06 tok/s | 81.06 tok/s |
| 端到端 | 40.77 s | 43.46 s |

端到端 0.938×，低于 1.00× 门槛。MTP 的解码收益不足以覆盖约 5.35 秒的首 token 延迟增量。按短测的速度门槛结束，三条真实任务尚未运行。80,000-token 仍是本协议下已通过速度及小批输出质量的最大实测输入。[复测配对数据](../results/kv-window-short-20261120/speed-128000-slots1-comparison.json)

### 约 64K 真实输入的双轮会话模拟（2026-11-20）

单并发、69,632-token FP8 KV 池、4 个 GDN 状态槽，使用相同量化 embedding、确定性 Triton 和 GDN prefill 图。比较同配置下 C1/D4 MTP 开关。用本地 LongBench v2 的印度能效融资报告和 BLIP 代码快照，各构造一个解释请求和一个短追问。两组首问前清缓存。追问沿用实际生成的首问回复，保留前缀。每侧每轮只跑一次，贪心采样，输出上限 128 token。

| 请求 | 输入 token | 缓存命中 token | 实际输出 token | 冷或热 TTFT（无 → 有 MTP） | 端到端（无 → 有 MTP） |
|---|---:|---:|---:|---:|---:|
| 报告问答 | 66,420 | 0 | 82 | 10.907 → 11.543 s | 12.521 → 12.652 s |
| 报告页码追问 | 66,534 | 66,368 | 3 | 0.116 → 0.122 s | 0.154 → 0.181 s |
| 代码解释 | 64,823 | 0 | 105 | 10.300 → 11.061 s | 12.384 → 12.107 s |
| 代码配置字段追问 | 64,959 | 64,768 | 5 | 0.113 → 0.122 s | 0.189 → 0.213 s |

四对输入 hash 匹配，输出均逐 token 一致且零回退。报告首问选 D，描述贷款汇集与 OEM 合作，追问答第 60 页。代码首问正确说明 `BertSelfAttention` 的 query/key/value 来源，追问答 `attention_probs_dropout_prob`。四次请求端到端合计 25.248 → 25.153 秒（1.004×），两次极短追问均有小幅增延迟。该样本中前缀复用将追问新计算范围缩到 166 和 191 token。无 MTP 和 MTP 侧启动前整卡已占用分别为 2,958 和 3,294 MiB。[原始配对与完整输出](../results/kv-window-short-20261120/real-64000-comparison.json)
