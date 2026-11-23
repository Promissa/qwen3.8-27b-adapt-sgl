# Qwen3.8-27B-NVFP4 × RTX 5090 D v2（24GB）SGLang 适配计划

**日常 PC 场景边界更新（11 月 21 日）。** 本项目面向日常 PC 使用场景，无法保证桌面与推理服务的显存分离。按用户决定，正式放弃无桌面（切核显）队列、Windows Sysmem Fallback 强制策略认证及“同等物理显存下 ≥2× 上下文”指标；改为在带桌面波动（1–4 GiB）下显式钉死安全池水位，避免触发高 PCIe 换页降速，并集中推进组合配方的去确定性税/8K 冷 prefill 优化与质量配对闭环。

**四项本地修复复测（11 月 19 日）** 已验收 Docker、Claude Code 工具调用与单卡 256K 容量。各公开工程子集的逐题成绩和协议差异见[修复记录](docs/agent-benchmark-repair.md)。

**五项官方基准复测（11 月 19 日）。** 最终质量评测新增 Terminal Bench 2.1 (Terminus)、SWE-bench Pro、DeepSWE 1.1、Agents’ Last Exam 和 IFBench。按完整评测集分别报告覆盖率、分数与方法差异。IFBench 采用 Qwen 推荐采样与发布时的官方评分器；较早的 BFCL/LongBench 子集继续作为优化对照记录。[本次运行记录与复现协议](docs/paper-benchmarks.md)

**2026-11-19 起的质量验收**采用应用任务，GSM8K 作为历史数学回归记录。主指标选 BFCL 的官方 multi-turn 类别任务成功率，补充指标选 LongBench v2 的长文档与代码仓库题准确率。中文文档可选 LongBench v1 的中文任务，编码工作流可选本仓库 OpenCode 回放。各项独立报告。

以未采用待测优化的参考配方重建分数。固定数据和评分器 revision、题目 ID、工具解析器、提示模板、thinking 设置及输出预算，对参考、Balanced、Max-context 逐题配对。长上下文主比较采用三者均能完整容纳的题，另报 Max-context 扩展集和覆盖率。先用重复参考运行估计波动，再于查看候选结果前冻结可接受退化幅度。已接入两套官方评分流程，历史配对结果见 [执行记录](docs/status.md)。来源见 [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) 和 [LongBench](https://github.com/THUDM/LongBench)。

### 固定子集与运行预算

**11 月 19 日加速执行更新。** 取消本轮及最终评测的稳定性测试，包括长时间 soak。最终评测前，原定同条件重复四轮及以上的测速改为两轮，保留每轮原始值并报告算术平均。质量子集中的不同题目、三处检索深度和独立候选继续逐项覆盖。实验顺序为冷 prefill、热前缀、无损推测验证、普通解码剖析。后文早期方案和已完成记录保留作溯源。


此次加速筛查已完成。16K 池的 MTP 图内验证在两个真实提示的两轮定长比较中为 2.41×，1K/8K 合成负载为 3.06×/2.84× 端到端，均逐 token 核对。49K Balanced 的 GDN prefill 图捕获在三个检索深度的热 TTFT 两轮均值均低于 50 ms，8K 冷 TTFT 仍待优化。详细协议见[实测与范围](docs/profiling.md#11-月-19-日加速筛查实测)。

**上下文极限与 80% 长度组。** 单卡、单并发，在 MTP 图内验证、Balanced 的 GDN prefill 图捕获和关闭 radix 的 Max-context 三套既定配方中移除固定 token 池上限，保留各自显存比例、状态槽及注意力后端。关闭自动截断，用相邻长度请求实测服务的最大输入：上界必须因长度被拒绝，下一 token 以内的输入必须完整读入并生成 1 个 token。以成功的最大输入乘 4/5 向下取整作为附加长度，1K、8K 和新增组均清缓存、greedy、固定生成 1,024 token、测速两轮取算术平均。按每个加速器的开关做同权重配对，核对实际池和逐 token 输出。Max-context 单独作容量参照；桌面显存占用随启动变化。新增质量检查覆盖三个深度的精确检索和续问、与目标长度相差不超过 5% 的未裁剪 LongBench 原题（无题时放宽至 10% 并记录实际长度）。加速候选另做冻结的 16 道 LongBench 和 8 个 BFCL 会话，以官方判分、逐题得失及截断数判断应用效果。当默认比例下的上界请求超过原有 180 秒客户端时限时，保留日志，下调静态显存比例后重新测相邻长度。另固定 MTP 的 45,956-token 输入，给长文原题和 1,024 输出只预留所需空间，对 48,128 池做独立配对。此轮沿用前置筛查预算。[实测文件](results/context80-20261119)

本轮已测得 MTP / GDN / Max-context 最大成功输入分别为 57,445 / 102,828 / 155,300，新增 80% 测点为 45,956 / 82,262 / 124,240。45,956 输入采用 48,128 池的 MTP 配对达到 1.845×；近 80% 的原题绝对得分分别为 0/3、1/4、0/4。详细的测速、逐 token 核对及输出截断见[实测报告](docs/profiling.md#实测上下文极限与-80-输入)。

| 用途 | BFCL 多轮 | LongBench v2 | 运行条件 |
|---|---:|---:|---|
| 日常筛查 | 8 个完整会话 | 4 题 | 每个影响输出的候选 |
| 入选配方确认 | 32 个完整会话 | 16 题 | 含筛查集，仅补跑 24 + 12 个留出样本 |

在查看输出前固定数据版本，每层按 seed 42 对题目 ID 确定性抽样并归档清单。BFCL 按多轮子类别均衡抽取，保留完整对话与工具状态。LongBench 在长文档与代码仓库两类内，按参考和候选共同支持的完整提示长度分短、长两档，四格各取 1 题筛查、4 题确认。样本不足时记录缺口。81K 特有能力沿用单独的容量测试。

本轮在 48,640 输入 token 的筛选上限内找到 3 道代码仓库题。确认集保留这 3 题，并从同长度档的长文档题补足至 16 题。筛查仍各取四格中的一道。评分协议、具体题目、分别实测的完整输入长度及替补标记见 [本轮评测资料](results/continuation-20261119)。

逐题保存参考分数、输出、耗时及协议标识。同一参考权重、引擎、请求设置和评分协议下复用参考结果；改变参考路径时重新测量。留出题用于确认，不参与调参。出现新增失败先复测分歧题，再按需扩测对应类别。报告各基准分子/分母、配对变化和总墙钟时间，一次试跑后用各层实测耗时估算剩余时间。小样本用于识别明显退化；如需验证细微质量差异，另做样本量设计。

日期：2026-11-10　|　2026-11-12 执行更新：按用户3小时收尾要求完成缩减验证与配方交付，10项串行试验及最终独立审计均完成。balanced/max-context 实际检索输入为48,640/81,400 token；两套脚本已按通过配置更新。原始≥1.5×无损加速未达成，匹配物理显存条件下≥2×上下文、无桌面/回退策略仍未验证。原始指标保留；完整结果见 `docs/three-hour-delivery.md` 与 `docs/status.md`。

本文分两部分：第 1–9 节是把 cookbook 配方适配到 24GB；第 10–14 节是在适配基础上的优化提升。

**2026-11-12 用户更新：3小时内收尾。** 按用户“适量减少测试数量”的新要求，
从17:05至20:05（新加坡时间）执行有截止时间的串行验证：复用已完成的全量质量、
包络和两小时稳定性证据；新增候选/最终配置的 GSM8K 筛查改为64题，仍保持16k
生成上限；辅助质量改为每域6个PPL窗口和32道greedy题；两档1024输出基准各1次；
最终配置各做5分钟运行检查。原1319题、3次重复和2小时门槛保留为完整验收标准，
缩减结果明确标注范围，不据此宣称通过0.5pp质量或噪声门槛。旧长队列已取消并保留
434条原始预测。预算、实测耗时估计和执行记录见
`results/budget-3h-20261112/plan.json` 与 `docs/three-hour-delivery.md`。

## 1. 目标与结论先行

SGLang cookbook 的 Qwen3.8-27B 页面已经覆盖 **RTX 5090（32GB）**，但没有 24GB 的 5090 D v2。两张卡同为 SM120，可继承主要算子路径；但 **D v2 的显存位宽是 384 位，32GB 5090 是 512 位**，差异包含容量和带宽，不能要求相同的带宽受限解码速度。本机 NVML 已读到 384 位（`results/gpu-hardware-live.json`），也与 NVIDIA 的两张卡规格页一致。所以本项目不是"从零适配"，而是：

> 把 cookbook 的 32GB 配方重新定标到 24GB，产出一套实测过的启动命令、内存包络表和基准数据，格式与 cookbook 的 cell 对齐，可回馈上游。

主要的可行性风险是：**放不放得下**。按权重文件逐张量核算（见第 15 节），cookbook 默认的 RadixArk 权重常驻约 19.6GiB，而本机扣掉桌面占用后只有 20.6GiB 空闲，按初始预算不能直接照搬。cookbook 文中"NVFP4 权重约 16.5GB"的说法与文件实际大小不符，本计划初稿据此做的估算已在第 4、5 节更正。实际执行采用全线性层 NVFP4、纯文本和 BF16 状态；桌面仍占约 3.4GiB 时已通过一个容量点。把桌面移出独显仍需单独定标，不能把初始估算当作实测失败。另发现源码的 CLI allowlist 漏了 Qwen3.5 纯文本架构，已在独立 checkout 加一行修复，原 checkout 不动。新增的 opt-in GDN 精度与 MTP 生命周期补丁分别通过16例隔离一致性和24,576-token容量，但全模型 greedy 均未通过。阶段 1 仍是硬性的可行性闸门。

**完成标准**

| 级别 | 内容 |
|---|---|
| 必达 | 无推测解码配方稳定启动并服务；通过本文顶部的应用任务质量验收；给出上下文长度上限与 tok/s |
| 应达 | 至少一种推测解码配方可用（NGRAM 或 NVFP4 草稿的 DSpark）；工具调用 / 推理解析冒烟通过 |
| 争取 | DSpark 或 DFlash2 草稿模型配方；向 cookbook 提交 `rtx5090dv2` 硬件 cell |

## 2. 执行前盘点（当时已实测；当前运行状态见 docs/status.md）

| 项 | 值 | 影响 |
|---|---|---|
| GPU | RTX 5090 D v2，24455 MiB，驱动 617.42，CUDA 13.4 | SM120，有原生 FP4 |
| 已占用显存 | 3367 MiB（Windows 桌面，WSL2 直通） | 可用仅约 20.6GiB，是最大的可调杠杆 |
| 系统 | WSL2，内核 6.18，16 核，内存 30GB，磁盘余 825GB | 无 Docker，走 uv 路线；加载 22GB 权重内存够用但不宽裕 |
| Python | 系统 3.10.12，有 uv，未装 sglang / torch | 需新建 venv |
| 本地 sglang 源码 | `~/projects/sglang`，main @ c3c292b65a（11-07），`tokenizer_manager.py` 有未提交改动 | 可用于 editable 安装与上游 PR；注意那处本地改动 |
| PyPI sglang | 最新 0.5.21；cookbook 的 SM12x 数据测于 0.5.19 | 首选 0.5.21，0.5.19 作对照 |
| 已有模型 | 只有一个 GGUF（IQ4_XS），无 NVFP4 权重 | 需下载约 22GB |
| 相关旧项目 | `~/projects/qwen38-27b-rtx3090/bench/`（conc_ladder、seat_ttft、quality_battery、api_smoke 等） | 基准脚本优先复用，不重写 |

## 3. Cookbook 中可直接继承的内容

**32GB 5090 的基线配方**（NVFP4，无推测，单请求）：

```bash
sglang serve \
  --trust-remote-code \
  --model-path RadixArk/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --attention-backend flashinfer \
  --max-running-requests 1 \
  --cuda-graph-max-bs-decode 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 30000
```

**不变的规则**

- SM120 基线用 `--attention-backend flashinfer`。固定源码的 `trtllm_mha` 已包含 SM120 XQA decode；NVFP4 KV 候选明确用 flashinfer prefill（FP8 workspace）+trtllm_mha decode，不能照搬 SM100 的原生 NVFP4 prefill。MTP + flashinfer 需要 FlashInfer 高于 0.6.15.post1，否则推测解码改用 `--attention-backend triton`。
- 所有 cell 都显式写 `--kv-cache-dtype fp8_e4m3`（NVIDIA 导出版不声明 KV 方案，`auto` 会落到 BF16，KV 体积翻倍）。
- `--chunked-prefill-size 2048`，显存紧张时降到 1024 / 512。
- 解析器固定 `qwen3` + `qwen3_coder`（不是 hermes）。
- 采样：thinking 模式 temp 1.0 / top_p 0.95 / top_k 20。

**内存模型**（本项目所有定标都基于它）

| 量 | 数值 |
|---|---|
| GDN 状态槽 | float32 153.9MB，bfloat16 78.4MB |
| KV 每 token | fp8 32.8KB，bf16 65.5KB |
| 每请求状态槽数 S | `extra_buffer` 5，`extra_buffer_lazy` 4，`no_buffer` 3，`--disable-radix-cache` 1 |
| 推测验证项 D | EAGLE 4，DSpark 8，DFlash2 8；无推测时为 0；ReplaySSM 只有受支持的组合才为 0（本 GDN 的 DSpark/DFlash 组合被源码拒绝） |
| 平衡比例 | `ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)` |
| 显式钉死 | `--max-mamba-cache-size = 并发 × S`（会覆盖 ratio） |

32GB 卡上的参考值：无推测 0.90；EAGLE bf16 0.93 / fp32 0.94；DSpark 0.88–0.92 并钉 `--max-total-tokens 16384`；DFlash2 0.91 + prefill 1024。32GB 卡在 bf16 状态下 KV 池为 97,280 token。

## 4. 24GB 预算估算（待阶段 1 验证）

本节已按第 15 节的查证结果重算，取代初稿中基于"权重 16.5GB"的表。仍是推算，不是实测。

池预算的算式来自 SGLang 源码（`kv_cache_configurator.py` 的 `_profile_available_bytes`）：

```
池预算 = 加载后空闲显存 − 加载前空闲显存 × (1 − mem_fraction_static) − 多模态预留(0.1GiB)
```

假设：权重常驻量 = 文件内张量大小 + 约 1GiB（CUDA 上下文等，由 32GB 卡的 cookbook 数据反推）；`mem-fraction-static` 取 0.90；单请求、bf16 状态、S=4（状态池约 0.3GiB）；KV 为 fp8，每 token 32.8KB。

| 权重 | 常驻估算 | 桌面占着（空闲 20.6GiB） | 桌面移走（空闲约 23.5GiB） |
|---|---|---|---|
| RadixArk，含视觉塔 | 约 20.6GiB | 无法启动 | 池约 0.5GiB，基本不可用 |
| RadixArk，纯文本 | 约 19.8GiB | 无法启动 | 池约 1.4GiB，约 35k token |
| gittensor，含视觉塔 | 约 17.7GiB | 池约 0.8GiB，约 17k token | 池约 3.4GiB，约 100k token |
| gittensor，纯文本 | 约 16.8GiB | 池约 1.7GiB，约 45k token | 池约 4.3GiB，约 130k token |

由此得到四条定标原则：

1. **主选权重必须是全线性层 NVFP4 的导出版**（gittensor 这一类），RadixArk / NVIDIA 的混合精度版在 24GB 上没有实用空间。
2. **把桌面移出独显接近必选项**，本机有核显可用（见第 15 节）。
3. **纯文本场景加 `--language-model-only`**，省掉约 0.9GiB 的视觉塔；本源码版本需要一行 CLI allowlist 修复。
4. **bfloat16 状态是默认项**；FP32状态现已通过容量/API/检索对照，仍需独立质量/速度评估。BF16-head和整体FP8权重不作为首选，是否可用依实际预算判定。

另有一个限制：对混合 GDN 模型，SGLang 可能把运行时预留下限抬到 `512 + 1.5 × max(chunked_prefill_size, 2048)` MB，约 3.5GiB（`pre_capture_activation_reserve_mb`）。若该分支生效，上表各项还要再少约 1GiB，而且把 prefill chunk 降到 2048 以下不会减少这部分预留。该分支何时生效没有从代码里完全确认，需启动日志验证。

### 4.1 已测纯文本点（不是最大值）

`results/probes/baseline-near-limit-088/probe.json` 全部通过：桌面仍连接独显，
fraction .88、BF16 状态、lazy cache、chunk 2048。启动时 GPU 已用 1149MiB，
不同于最初约 3.4GiB 的桌面快照，不能把这些点当作恒定桌面占用下的配对测量。

| 项 | 实际日志 / 源码核算 |
|---|---|
| 加载前 / 后空闲 | 22.10 / 4.64 GiB；加载阶段差值 17.46 GiB |
| 状态池 | 11 个逻辑槽，实际分配 12 个（含 padding 槽）；SSM 0.84375GiB，conv 0.03296GiB |
| FP8 KV | 36,512 token；每 token 32,768 字节，名义 payload 约 1.1143GiB |
| 分配池后空闲 | 2.63GiB |
| 图捕获 | prefill 日志差值 +0.51GiB、decode −0.09GiB；可用显存受其他进程影响，不能把差值当成精确独占分配量 |
| 工作负载 | 接近池上限的容量请求、8192/1024、36,000 输入预算的三深度自然检索与前缀命中均通过 |

逐项字节核算见 `results/memory/baseline-near-limit.json`。上游预算已经计算
padding 状态槽，不需要为此补丁。纯文本不收取多模态缓存预留；前表统一扣
0.1GiB 是初始估算的简化。3712MiB 预捕获下限所在的 post-capture sizing 开关
在本次启动中为 false，因此没有进入该分支。Windows NVAPI 设置 sysmem 策略
返回 −160，未保存设置；仍待用户侧确认，当前数据不能认证纯物理显存 OOM 边界。

## 5. 权重选型

顺位已按第 15 节的查证结果调整（初稿以 RadixArk 为主选）。

| 顺位 | 仓库 | 文件大小 | 理由 |
|---|---|---|---|
| 主选 | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` | 17.92GB | 注意力和 GDN 投影也是 NVFP4，是唯一在 24GB 上有实用余量的现成权重。不含内置 MTP，不能直接以自身权重启动 EAGLE；单独指定参考 checkpoint 的量化 MTP 已有源码依据，0004提前共享后容量已通过24,576-token，但严格greedy0/8，共享后的无损精度未通过。第三方导出，质量需自行复核 |
| 质量对照 | `RadixArk/Qwen3.8-27B-NVFP4` | 21.92GB | cookbook 默认，注意力和 GDN 投影为 FP8，自带 BF16 的 MTP 头。只在桌面移走 + 纯文本下勉强能跑，用来对比 gittensor 版的质量损失 |
| 参考 | `nvidia/Qwen3.8-27B-NVFP4` | 21.95GB | 与 RadixArk 同一量化层图，体积相同，不单独测；其卡片上的 BF16 对比精度表可作参照 |
| 候选 | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | 20.58GB | QAT 训练，compressed-tensors 格式，SGLang 代码里有对应的稠密 W4A4 路径；但体积比 gittensor 大 2.7GB，优先级低 |
| 排除 | `unsloth/Qwen3.8-27B-NVFP4` | 23.44GB | 体积最大 |

三个 ModelOpt 权重的 `embed_tokens` 都是未量化的 BF16（2.54GB），这是下一步最值得收回的一块，见第 11 节 B 层。

## 6. 阶段划分

### 阶段 0：环境（闸门：`sglang serve --help` 可用，FP4 GEMM 后端可导入）

- uv 建 Python 3.12 venv，`uv pip install --prerelease=allow sglang`，记录 sglang / torch / flashinfer 版本。
- 权重下载到 ext4（不要放 `/mnt/c`），记录 revision。
- Windows 侧把 NVIDIA 的 **CUDA Sysmem Fallback Policy 设为 Prefer No Sysmem Fallback**。否则显存超限会静默溢出到共享内存，表现为速度骤降而不是 OOM，所有包络测量都会失真。
- 检查 `.wslconfig` 内存上限；确认是否能把显示输出切到核显。
- 写 `scripts/probe.sh`：启动、等待 READY、抓取日志里的 `max_total_num_tokens`、`max_running_requests`、状态池大小、峰值显存。

### 阶段 1：能否启动（硬闸门）

- 前置：桌面移出独显（显示器接主板输出）。
- 用第 3 节基线命令，模型换成 gittensor 权重，加 `--language-model-only --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy --chunked-prefill-size 2048`，从 `--mem-fraction-static 0.88` 起步。
- 从启动日志记录：加载前后空闲显存、实际运行时预留、KV token 数、状态槽数，回填第 4 节的表，替换掉估算值。重点确认 3.5GiB 的预留下限是否生效。
- **通过**：启动成功，单请求 8192 入 / 1024 出可完成。
- **失败的降级顺序**：S=1（`--disable-radix-cache`）→ 调高 `mem-fraction-static` → 0.5.19 对照排查回归。
- 通过后再去掉 `--language-model-only` 试一次，确定带视觉塔的配方是否可用。

### 阶段 2：内存包络扫描（核心产出）

二分搜索每个组合下能启动且跑完长 prefill 的最大 `mem-fraction-static`，并记录对应的 KV token 数。

| 维度 | 取值 |
|---|---|
| 桌面占用 | 有 / 无 |
| 状态 dtype | bfloat16（另测一个FP32对照组合；实际容量可行，原先“确认不可行”的假设不成立） |
| 缓存策略 | extra_buffer / extra_buffer_lazy / no_buffer / 关闭 radix cache |
| prefill chunk | 2048 / 1024 / 512 |
| 并发 | 1，可行则试 2 |

每个点的判据：启动 + 一次接近满上下文的 prefill + 一次 8192/1024 请求，三者全过才算数。注意 cookbook 的教训：池能放下不代表 prefill 的激活能放下。

产出两个命名配方：**max-context**（上下文最大化）和 **balanced**（保留前缀缓存，日常 agent 使用）。

### 阶段 3：推测解码

按显存代价从低到高依次尝试，每一步都是独立闸门：

顺序已按第 15 节调整：主选权重不含 MTP 头，EAGLE 不再是首选。

1. **NGRAM**（`--speculative-algorithm NGRAM`）：SGLang 自带，用已生成的 token 做 n-gram 草稿，不需要草稿模型，不占额外权重显存。固定源码的 CUDA 路径保留 overlap，但关闭 mixed chunked prefill（CPU 路径另行关闭 overlap）。默认 12 个 draft 在本机状态预算失败；降为 4、钉 4 状态槽后容量通过，但 8 个 greedy prompt 均不与基线逐 token 相同，未过无损闸门。
2. **DSpark，NVFP4 草稿**（`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`，1.40GB，加 `--speculative-draft-model-quantization modelopt_fp4`）：需要 `--max-mamba-cache-size ≥ 4 × 并发 + 4`。原版 `RadixArk/Qwen3.8-27B-DSpark` 是 BF16 的 3.71GB，放不下。
3. **DFlash2**（`incoai/Qwen3.8-27B-DFlash2`，draft tokens 8）：32GB 上最快的配方（TPOT 中位 4.92ms，接受长度 4.29），但原版是 BF16 的 3.85GB，要先自行量化（第 11 节 B 层）。
4. **EAGLE + ReplaySSM**：RadixArk 自带 BF16 MTP（0.85GB）。其 attention/MLP 已实际离线转为 W4A16 NVFP4，MTP 张量共314,297,372字节，省535,101,412字节。固定源码允许独立的 Qwen draft checkpoint，“主选 target + 外部量化参考 MTP”已实测：late-sharing在.92失败，0004提前共享后通过24,576-token容量，但严格greedy0/8，无速度测量。独立native-profile与packed-GDN对照排在包络扫描后。

每个配方记录：能否启动、上下文上限、接受长度、tok/s。放不下的要写明差多少，这本身就是结论。

### 阶段 4：正确性与质量

- 每个最终配方按[固定子集与运行预算](#固定子集与运行预算)完成 48 个配对样本。已完成日常筛查的配方仅补跑 36 个留出样本。
- 工具调用（`tool_calls` 结构化返回）、`reasoning_content` 分离、`enable_thinking=false`、`reasoning_effort`。
- 图像输入冒烟，并观察视觉请求对显存峰值的影响。
- 长上下文：贴近上限的 prefill、多轮前缀缓存命中。
- 推测解码无损性：temp 0 下与无推测输出逐 token 对比。
- 留意社区反馈的"thinking 循环 2–5 万 token"问题，是否在本配置复现。

### 阶段 5：性能基准

沿用 cookbook 的协议，保证可比：

- `python -m sglang.benchmark.serving --flush-cache`，random 数据集，`--random-range-ratio 1`，并发 1。
- 两个档位：ISL 1024 / OSL 1024（与 cookbook 基准表对齐）和 ISL 8192 / OSL 1024（与 SM12x 验证包络对齐）。
- 指标：TTFT、TPOT 中位数、tok/s、接受长度、峰值显存。
- 参照点：第三方卡片报告 32GB 5090 上 SGLang 无推测约 86–88 tok/s，内置 MTP 约 137 tok/s。D v2 的 384 位显存接口更窄，不能直接用 32GB 卡的速度要求本机；较低的 tok/s 需要结合带宽、trace、时钟和 sysmem 设置判断，不能单凭差距认定回退或后端错误。
- 真实负载补充：复用旧项目的 `prompts_real.jsonl`、`conc_ladder.py`、`seat_ttft.py`。

### 阶段 6：稳定性（11 月 19 日取消，以下为历史设计）

- 2 小时混合负载 soak（长短 prefill、图像、工具调用交替），监控显存曲线与 WSL 内存。
- 冷启动耗时、重启后恢复、Windows 侧开关其他占显存程序时的行为。

### 阶段 7：交付

- `docs/recipe.md`：按 cookbook 页面结构写（部署命令、配置要点、内存表、已知问题）。
- `scripts/serve-*.sh`：每个命名配方一个启动脚本。
- `results/`：包络表、基准 JSON、GSM8K 结果，附 sglang / flashinfer 版本和权重 revision。
- 可选上游 PR：在 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx` 增加 24GB 硬件项和对应 cell，Configuration Tips 增补 24GB 段落。提交前跑 `docs/scripts/check_cookbook_configs.mjs`。

## 7. 仓库结构

```
PLAN.md
scripts/   setup-env.sh  download.sh  probe.sh  serve-max-context.sh  serve-balanced.sh  serve-eagle.sh
bench/     envelope_sweep.py  run_serving_bench.sh  （其余从 qwen38-27b-rtx3090/bench 复用）
results/   envelope.csv  bench/*.json  gsm8k/*.json  env.txt
docs/      recipe.md  gotchas.md
```

## 8. 风险

| 风险 | 可能性 | 应对 |
|---|---|---|
| 主选权重在桌面占用下无法启动 | 中 | 阶段 1 的降级顺序；最坏情况用 gittensor 权重 |
| WSL2 sysmem 回退掩盖 OOM | 高（默认行为） | 阶段 0 关闭回退；基准中用 tok/s 参照点交叉检查 |
| 推测解码全部放不下 | 中 | 以无推测配方为必达交付；记录缺口作为结论 |
| 0.5.21 相对 0.5.19 有 SM120 回归 | 低 | 0.5.19 对照环境 |
| FlashInfer 首次启动需 JIT 编译 SM120 FP4 GEMM（需要 nvcc 和 CUDA 头文件，吃内存） | 中 | 阶段 0 提前触发；内存不足时设 `MAX_JOBS=2` |
| 30GB 主机内存加载 22GB 权重时吃紧 | 低–中 | 调大 `.wslconfig`，加载期间关闭其他进程 |
| 第三方卡片数据不可复现 | 中 | 只当参照，结论全部以本机实测为准 |

## 9. 待你确认的两个决定

1. **显示输出能否切到核显？** 已查证：能。本机 CPU 是 Ryzen 7 9700X，核显 AMD Radeon Graphics 已启用。把显示器接到主板输出即可，这一步需要你动手。接近必选项，见第 4 节。
2. **主要使用场景是单用户 agent（上下文优先）还是追求速度（推测解码优先）？** 两者在 24GB 上互相争显存。计划默认按单用户 agent、上下文优先来排优先级。这是你的取舍，无法查证。
3. **是否需要图像输入？** 不需要的话加 `--language-model-only`，省约 0.9GiB。计划默认按纯文本排。

---

# 第二部分：适配之上的优化提升

第一部分的终点是"cookbook 配方在 24GB 上跑通并定标"。第二部分的目标是**超过 cookbook 配方**：同一张卡上拿到更长的上下文和更高的单流速度，且质量不退。

## 10. 思路：先收回显存，再换成速度

24GB 上显存是唯一的硬约束，而最大的速度杠杆（推测解码）恰恰最吃显存。所以优化顺序是固定的：

1. 先从权重、KV、状态池里**收回显存**；
2. 用收回的显存**放下更好的草稿模型**，或换成更长上下文；
3. 最后才做解码路径上的**微观提速**。

参照量级（32GB 5090 上的第三方自测，仅作方向参考）：无推测约 88 tok/s，内置 MTP 约 137，DSpark 约 180。也就是说，在 24GB 上把推测解码"放进去"本身就值 1.5–2 倍，远大于任何单点内核优化。

**暂定目标**（相对第一部分产出的基线，阶段 8 剖析后修正）：

| 指标 | 目标 |
|---|---|
| 上下文上限（同等桌面占用） | ≥ 2 倍 |
| 单流 tok/s | ≥ 1.5 倍，争取 2 倍 |
| 质量 | 应用质量按[固定子集与运行预算](#固定子集与运行预算)验收；推测解码在 temp 0 下逐 token 无损 |
| 多轮对话后续轮 TTFT | 前缀缓存命中时 ≤ 1.5s |

## 11. 优化项清单

按"改动层级"分四层，层级越高成本和风险越大。显存收益一栏均为估算，以实测为准。

### A 层：只改启动参数和环境变量（零代码）

| 项 | 做法 | 预期 |
|---|---|---|
| 每请求少占一个状态槽 | `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` | S 减 1，并发 2 时更有意义 |
| 状态检查点放主机内存 | `--enable-hierarchical-cache` + `SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT=1` | 前缀缓存的状态不再占 GPU 槽；主机只有 30GB 内存，需限额 |
| CUDA graph 显存 | `SGLANG_ENABLE_GRAPH_POOL_PRECARVE`、`--cuda-graph-max-bs-decode 1` | 压缩非池开销，直接抬高可用 mem-fraction。（`SGLANG_MEMORY_SAVER_CUDA_GRAPH` 已查证只作用于多卡 all-reduce，单卡无效，已删去） |
| NVFP4 KV cache | `--kv-cache-dtype nvfp4 --prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha` | 上游已支持 SM120，不需要打补丁。持久 packed KV 加 scale 为18,432字节/token；当前 FlashInfer prefill 路径另需跨层共享 FP8 工作区2,048字节/token，实际定标20,480字节/token。同 KV 字节预算名义容量倍率约1.6，而非仅按持久数据估计的1.78；真实容量/质量仍需实测，见第15节与 `results/memory/kv-nvfp4-full-storage-accounting-20261112.json` |
| GDN 的 FlashInfer 后端 | `--linear-attn-decode-backend flashinfer`、`--linear-attn-prefill-backend flashinfer` | 默认只在 SM100 自动启用；SM120 上参数检查不拦，但内核是否支持只能实测 |
| FP4 GEMM 后端 A/B | `--fp4-gemm-backend flashinfer_cutlass`（SM120 的 auto 结果）对比 `flashinfer_cudnn`（需 CUDA 13 + cuDNN 9.15+） | 单步解码时间，胜者未知 |
| 验证步开关 | 三个原候选均已追到消费者：`SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN` 与 `SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR` 对应 EAGLE 路径；`SGLANG_OPT_FUSED_KDA_VERIFY` 仅 KDA。本 GDN 的 DSpark 组合不激活这些分支，已标记为不适用；EAGLE 可行后另做有实际消费者的对照 | 不能把无效开关的噪声当提速；详见 `docs/source-audit.md` |
| 注意力后端 A/B | flashinfer 对比 triton（推测验证步） | 同上 |
| 推测验证的确定性控制 | `--enable-deterministic-inference`，并按固定源码要求使用 Triton linear-attn prefill；FlashInfer 确定性组还需关闭 radix cache。每组先捕获自身无推测控制，再与同组推测解码逐 token 对比 | 实际 BF16 GDN gate 投影原生路径14/56例逐bit一致，ATen invariant模式56/56一致；普通NVFP4线性层160/160一致。不是新内核，也不是全模型无损结论；耦合配置须有自身控制、完整质量和速度证据 |
| 推测参数扫描 | EAGLE 的 steps / draft-tokens，DSpark 的 block-size | 找接受长度与每步成本的最优点；cookbook 提示 draft-tokens 4 可能是接受长度的上限 |
| 状态精度 | bf16 对比 fp32（cookbook 记录过 fp32 反而更快的组合） | 显存够时可换速度 |

A 层全部做完再进入下一层，因为它几乎没有成本，而且会改变后面各项的收益排序。

### B 层：只动权重文件（离线处理，不改引擎）

| 项 | 做法 | 预期 |
|---|---|---|
| **embedding 表量化** | 把 `embed_tokens`（BF16，2.54GB）转成 NVFP4。通用引擎已有 `ModelOptNvFp4EmbeddingMethod`（只做 gather 时反量化），但 Qwen3.5 backbone 未把 quant_config/prefix 传给 embedding，实测 packed/BF16 形状不匹配；需要 `0002-qwen35-quantized-embedding-dispatch.patch` 接线，同时离线改权重文件。embedding 是纯查表，不需要校准数据，可在 CPU 上离线转换 | 文件节省约1.7GiB；全量 GSM8K1281/1319，下降0.3033pp，通过0.5pp闸门；辅助195/200、PPL7.86932。最终组合已完成用户授权的64题/缩减辅助/五分钟验证；原始全量认证仍未建立，不能把全部 GPU 占用差归因于 embedding |
| 草稿模型量化 | DSpark 直接用现成的 `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`（1.40GB）；DFlash2 原版 BF16 3.85GB，自行量化到 NVFP4 | 这是 DFlash2 能否进 24GB 的决定因素，省约 2.5GB |
| 主干量化程度的取舍 | 主选的 gittensor 版已是全线性层 NVFP4；与 RadixArk 版（注意力和 GDN 投影为 FP8，大 4GB）对比质量 | 确认为显存付出的质量代价可接受 |
| QAT 权重评估 | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`（20.58GB，自报 GPQA 0.909 对比 BF16 的 0.914） | 代码里有加载路径，但体积比 gittensor 大 2.7GB，优先级低 |
| MTP 头量化 | 已将参考 checkpoint 的七个 attention/MLP 矩阵转为 W4A16 NVFP4，context fc 与 norms 留 BF16；849,398,784 →314,297,372字节，省约0.498GiB。真实 Qwen constructor/loader fixture 与不变张量检查通过 | 除参考 target 外，也准备了主选 target + 独立 MTP 的候选；0004提前共享后GPU容量通过24,576-token；严格greedy0/8，无损、接受率与速度未通过 |

### C 层：改引擎（在独立 `engine/sglang` checkout 上打补丁，原目录不动）

| 项 | 依据 | 预期 |
|---|---|---|
查证后这一层大幅缩水：初稿列的七项里，四项上游已有实现（已移到 A、B 层或第一部分），一项不成立。

| 项 | 查证结论 | 剩余工作 |
|---|---|---|
| NVFP4 KV cache 支持 SM120 | 上游已实现（PR #21601），有 5090 上的 CI | 无需补丁，移到 A 层做配置实验 |
| embedding 表量化 | 通用 NVFP4 gather 已有，但 Qwen3.5 backbone 漏传量化配置和前缀 | 接线补丁已通过真实 checkpoint 的容量、API、84,787 输入预算检索；全量 GSM8K 通过0.5pp闸门、辅助质量完成；最终组合已完成缩减验证，未建立原始全量认证；不是新的反量化内核 |
| 视觉塔按需卸载 | 基类实现支持，但 CLI 架构 allowlist 漏了 Qwen3.5，实测在加载前拒绝 | 独立 checkout 加一行架构名后已通过文本容量测试；是可行性修复，不是新的卸载实现 |
| GDN prefill 快速路径 | 参数层面不拦 SM120，内核支持情况从 SGLang 代码看不出 | 移到 A 层做参数实验 |
| 小 top-k 免排序采样 | SGLang 采样器用的是 FlashInfer 的 top-k / top-p 重归一化和 `torch.topk`，没有全词表排序 | 不成立，删去 |
| 上下文查找式草稿 | SGLang 有 NGRAM 推测算法，思路相近，但它是独立算法，不能与 DSpark / DFlash2 叠加 | 先在阶段 3 测 NGRAM 单独的效果；"查找 + 模型草稿混合"才是真正要写的补丁，视 NGRAM 的结果决定做不做 |
| 精简草稿词表 | `--speculative-token-map` 的实际加载、裁剪 head、索引映射都在 EAGLE worker；DSpark / DFlash 无消费者，缩词表的 rejection sampling 另有显式拒绝 | Qwen MTP + packed NVFP4 head 的权重与 scale 接线仍需独立验证；主选权重不走 MTP，优先级低 |

最初的 C 层清单只留下混合查找草稿候选；执行中发现三个有证据的引擎问题：0003使GDN的beta cast/每步BF16状态持久化在16例隔离测试中逐bit一致；0004把Qwen MTP的既有权重共享提前到target池定标之前，实际修复.92下的容量失败；0005统一tree convolution与顺序decode的累加顺序，16例真实CUDA输出与状态均逐bit一致。0003+0005的全模型NGRAM仍为0/8，0004的MTP严格greedy也为0/8；三个实验开关默认关闭，不计作全模型无损或速度收益。新增0006通过实际缓存KV原生回放定位并修复捕获样本的attention tile差异，完整8提示严格检查仍为0/8，默认关闭。其余初始候选仍是不改引擎的实验。旧项目（`~/projects/qwen38-27b-rtx3090`）的补丁都是针对 vLLM 的，这次查证说明其中多数在 SGLang 里已有对应实现。

### D 层：使用方式

- 前缀缓存策略：agent 场景固定系统提示并预热，比较 `extra_buffer_lazy` 与 `no_buffer` 在多轮对话下的命中率和 TTFT。
- `reasoning_effort` 默认档位：thinking token 是单用户场景下最大的时间开销，按任务类型给出推荐档位。
- 并发 2 的可行性：单用户 agent 常有并行子任务，显存收回后重新评估。
- 接入 OpenCode / Claude Code 等 harness 的实测配置（cookbook 第 3 节已有模板）。

## 12. 方法

- **先剖析再动手**：对基线配方做一次单步解码的耗时分解（主干 GEMM、GDN、注意力、lm_head、采样、草稿、调度开销），按占比决定 C 层的先后，占比小的项直接砍掉。
- **一次只改一项**，固定协议（第 6 节阶段 5 的两档负载 + 真实 prompt 集），每个点重复 3 次，先测出噪声底，低于噪声的收益不计入。
- **三个指标同时记**：显存（峰值、KV token 数、状态槽数）、速度（TTFT、TPOT、tok/s、接受长度）、质量。
- **质量闸门分两级**：
  - 无损类改动（推测解码、采样路径、缓存策略）：temp 0 下与基线逐 token 一致。
  - 有损类改动（任何再量化、KV 降精度、状态降精度）：先按[固定子集与运行预算](#固定子集与运行预算)完成 12 个应用样本筛查，长上下文检索沿用独立容量测试。KV 和状态的精度损失在长上下文下才显现，短题测不出来。
- **收益台账**：`results/ledger.md`，每行一项优化，记录它单独带来的显存 / 速度 / 质量变化和累计值，仿照旧项目 README 的"what each step buys"。没有收益的项也要记，避免重复试。

## 13. 阶段划分（接第 6 节）

| 阶段 | 内容 | 闸门 / 产出 |
|---|---|---|
| 8 | 剖析基线；修正第 10 节目标 | 单步耗时分解、显存分解 |
| 9 | A 层全部项逐一 A/B | 新的"调优后基线"配方 |
| 10 | B 层：量化草稿模型，重跑阶段 3 的三种推测解码 | 确定 24GB 上最快的无损配方 |
| 11 | B 层：更激进主干量化与 QAT 权重评估 | 过质量闸门才采纳 |
| 12 | C 层：按阶段 8 的占比和可行性排序，逐项做 | 每项一个独立补丁 + 台账一行 |
| 13 | D 层与最终组合验证：应用质量与性能基准 | 最终配方、台账、上游 PR 清单 |

阶段 12 是开放式的，按台账的边际收益决定何时停。查证后建议的总体起手顺序：embedding 量化（B 层，收益最确定）→ NGRAM 与 NVFP4 草稿的 DSpark（阶段 3）→ NVFP4 KV 实验（A 层，质量风险最高，放最后）。

## 14. 第二部分的风险

| 风险 | 应对 |
|---|---|
| 多项有损优化叠加后质量悄悄下滑 | 每叠加一项都跑质量闸门，并保留"仅无损优化"的配方作为对照 |
| 引擎补丁随 SGLang 升级失效 | 固定一个基准 commit；补丁保持小而独立；能上游的尽早提 PR |
| NVFP4 KV 质量不达标（权重里没有 KV scale，上游 CI 的阈值也很低） | 只作为可选配方，过不了长上下文质量闸门就放弃 |
| 主选权重是第三方导出，质量未经独立验证 | 阶段 4 与 RadixArk 版做同协议对比 |
| 主机内存 30GB 限制了 hicache | 给 hicache 设上限；必要时调 `.wslconfig` |
| 本地 sglang 仓库已有未提交改动 | 开工前先确认那处改动的来历，另开分支做补丁 |

## 15. 待定问题查证记录（2026-11-10）

方法：读本地 SGLang 源码（main @ c3c292b65a）、远程读取 HF 权重文件的 safetensors 头（只取张量名、精度、形状，未下载权重）、查询本机硬件。没有启动过服务，凡是需要实际运行才能回答的都归在 15.3，未作处理。

### 15.1 已有明确答案

| 问题 | 答案 | 依据 |
|---|---|---|
| 显示输出能否移出独显 | 能。CPU 是 Ryzen 7 9700X，核显 AMD Radeon Graphics 已启用。当前独显上挂着 explorer、Firefox、Steam、微信、MuMu 模拟器等约 30 个进程 | Windows 显卡列表、`nvidia-smi.exe` |
| `mem-fraction-static` 按总显存还是空闲显存 | 按**加载前的空闲显存**。池预算 = 加载后空闲 − 加载前空闲 × (1 − 比例) − 多模态预留。桌面占用已自动扣除，不会重复计算 | `kv_cache_configurator.py::_profile_available_bytes`、`bootstrap.py::measure_pre_model_load_memory` |
| 权重实际占多少 | RadixArk 版 21.92GB：MLP 9.63、GDN 投影 5.59（FP8）、embedding 2.54（BF16）、注意力 1.68（FP8）、视觉塔 0.92、MTP 0.85、lm_head 0.72。gittensor 版 17.92GB：GDN 3.17 和注意力 0.94 降为 NVFP4，无 MTP，其余相同 | safetensors 头逐张量统计 |
| cookbook 的"NVFP4 权重约 16.5GB"是否可信 | 不可信，与文件大小不符。用 cookbook 自己的 32GB 实测（0.90 下 KV 97,280 token）反推，常驻量也在 22GiB 左右 | 同上 + 池预算算式 |
| 无推测时 MTP 头是否占显存 | 不占。主模型加载时跳过所有 `mtp` 张量，只有开推测解码时草稿 worker 才加载 | `models/qwen3_5.py` 的 `load_weights` |
| 视觉塔是否占显存、能否关 | 默认加载（0.92GB，另预留 100MB 多模态缓存）。模型实现支持 `--language-model-only`，但原 CLI allowlist 漏了 `Qwen3_5ForConditionalGeneration`，实测启动被拒。独立 checkout 补上该架构后已启动并通过容量与文本功能测试，代价是不能输入图像 | `models/qwen3_vl.py:1340`、`configs/model_config.py`、`patches/0001-qwen35-language-model-only.patch`、`results/probes/baseline-capacity-088/` |
| RadixArk 版 MTP 头的精度 | BF16，0.85GB | safetensors 头 |
| 草稿模型大小 | RadixArk DSpark 3.71GB（BF16）；gittensor DSpark-NVFP4 1.40GB；DFlash2 3.85GB（BF16） | HF 文件列表 |
| NVFP4 KV 在 SM120 上是否要打补丁 | 不用。上游已支持，配置是 `--kv-cache-dtype nvfp4`，prefill 用 flashinfer（经 FP8 工作区反量化），decode 用 `trtllm_mha` 的 XQA 实现 | `arg_groups/kv_cache_hook.py`、`trtllm_mha_backend.py`、`test_llama8b_nvfp4_kv_cache_sm120.py` |
| SGLang 是否支持量化的 embedding | 通用 NVFP4 embedding 已有；Qwen3.5 backbone 需传入量化配置与 prefix，补丁后真实加载/容量/检索通过。要求 embedding 与 lm_head 不共享权重，Qwen3.8 满足 | `modelopt_quant.py::ModelOptNvFp4EmbeddingMethod` |
| SGLang 有没有查找式草稿 | 有 NGRAM 算法，不需要草稿模型；固定源码 CUDA 只强制关闭 mixed chunked prefill；CPU 才关闭 overlap | `speculative/ngram_worker.py`、推测解码文档 |
| 采样器是否全词表排序 | 否，此优化项不成立 | `layers/sampler.py` |
| `SGLANG_MEMORY_SAVER_CUDA_GRAPH` 对单卡有没有用 | 没有，只在多卡 all-reduce 里读取 | `distributed/` 下的三处引用 |
| FlashInfer 版本是否满足 MTP 的要求 | 满足。当前 main 固定 `flashinfer_python[cu13]==0.7.3`，高于 cookbook 要求的 0.6.15.post1。PyPI 的 0.5.21 固定的版本未单独核对 | `pyproject.toml` |
| QUASAR 权重 SGLang 能否识别 | 格式上能：它是 compressed-tensors 的 `nvfp4-pack-quantized`，代码里有稠密层 W4A4 路径（要求 SM100 及以上，SM120 满足） | `compressed_tensors.py` |
| 本地 sglang 仓库那处未提交改动是什么 | `tokenizer_manager.py` 里 `_abort_all_until_unlocked` 的一个小修复（等已派发的请求结束再退出循环），与本项目无关。做补丁时另开分支即可 | `git diff` |

### 15.2 有部分答案，但结论需实测确认

| 问题 | 已知 | 未知 |
|---|---|---|
| NVFP4 KV 的质量 | 文档称其为实验性，并说明长上下文下误差会累积。上游 5090 CI 用 Llama-3.1-8B-NVFP4 跑 GSM8K，30 次均值 0.620，阈值设在 0.60。三个候选权重里都没有 `k_scale` / `v_scale`，SGLang 会回退到 1.0 | 0.62 相对同模型 FP8 KV 的基线差多少（仓库里没有对应数字）；scale 为 1.0 对 Qwen3.8 影响多大；XQA decode 与混合 GDN 模型、与推测验证能否配合。代码里明确 XQA 不支持 ragged verify |
| 运行时预留的下限 | 混合模型在某个分支下预留会被抬到约 3.5GiB，且与 prefill chunk 是否低于 2048 无关 | 该分支（`post_capture_kv_active`）在本配置下是否生效 |
| 权重常驻量比文件大多少 | 由 32GB 数据反推约多 1GiB，但反推依赖 cookbook 那个数字对应的是哪个导出版 | 实际值 |
| GDN 的 FlashInfer 后端能否用于 SM120 | 自动选择只对 SM100 开启；显式指定时参数检查只要求 CUDA 13+（本机 13.4 满足） | FlashInfer 内核本身是否支持 SM120 |
| 状态检查点放主机内存 | 需要 `--enable-hierarchical-cache`，与 int8 状态检查点互斥；每请求槽数降到 1 + 跟踪槽 | 30GB 主机内存下的实际占用和恢复延迟 |
| `--speculative-token-map` 是否对 MTP / DSpark 生效 | EAGLE worker 有加载、裁剪 head 和索引映射；DSpark/DFlash 无消费者 | packed NVFP4 head 的 scale 接线与实际 MTP 推理尚待验证；缩词表 rejection sampling 有显式拒绝 |

### 15.3 必须落地回答的问题（执行状态）

- 24GB 上各配方能否启动、实际 KV token 数和上下文上限。
- gittensor 相对 RadixArk 的单次匹配全量 GSM8K 为1285对1278，辅助 greedy192对190，PPL7.89022对7.73355；embedding 全量1281/1319通过0.5pp闸门，辅助195/200、PPL7.86932。独立验证见 `docs/status.md`；这些单次结果不替代最终组合验证。
- 各推测解码方案的接受长度和速度；NGRAM 是否支持混合 GDN 模型。
- FP4 GEMM 后端、注意力后端、验证步开关之间的 A/B 结果。
- 社区反馈的 thinking 循环问题是否在本配置复现。
- Windows 侧 CUDA Sysmem Fallback 当前的设置（从 WSL 里读不到，需在 NVIDIA 控制面板查看）。
- 使用场景的取舍（第 9 节第 2、3 问），这是你的决定。

## 参考

- NVIDIA 硬件规格（执行中更正位宽假设）：https://www.nvidia.cn/geforce/graphics-cards/50-series/rtx-5090-d-v2/ ，https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/ ；本机 NVML 原始值：`results/gpu-hardware-live.json`。

- SGLang cookbook Qwen3.8-27B 页（本地 `~/projects/sglang/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx` 及 `docs/src/snippets/configs/Qwen/qwen3.8-27b.jsx`）
- SGLang 量化文档：https://docs.sglang.io/docs/advanced_features/quantization
- 旧项目的优化记录：`~/projects/qwen38-27b-rtx3090/docs/optimizations.md`（vLLM + 3090，第 11 节 C 层的思路来源）
- SGLang 源码中的相关开关：`python/sglang/srt/environ.py`（`SGLANG_OPT_MAMBA_*`、CUDA graph 相关）、`srt/layers/quantization/kvfp4_tensor.py`
- 模型卡：https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4 ，https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 ，https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 ，https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4
