# Qwen3.8-27B 公开基准复测

**四项本地修复复测（11 月 19 日）** 已验收 Docker、Claude Code 工具调用与单卡 256K 容量。各公开工程子集的逐题成绩和协议差异见[修复记录](agent-benchmark-repair.md)。

## 官方参照与本地覆盖

Qwen 的 27B 官方模型卡公布了五项分数，其中两项注明 Claude Code、采样和上下文窗口。可核实的 27B 发布材料没有逐项给出全部运行清单。Qwen3.8-Next 的架构论文适用于另一模型。下表“官方”列来自 27B 模型卡。[固定版本模型卡](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md)

| 基准 | 官方 | 本地协议与覆盖 |
|---|---:|---|
| Terminal Bench 2.1 (Terminus) | 73.0 | 官方注册表列出 89 项。当前缺少任务容器运行时。全量包含要求代理执行漏洞利用的任务，本次范围只允许常规工程任务。未生成轨迹或全量分数。[任务清单](https://github.com/harbor-framework/terminal-bench-2-1/blob/main/tasks/dataset.toml) |
| SWE-bench Pro | 61.7 | Claude Code，温度 1.0，top-p 0.95，256K 上下文。Qwen 还修订了有问题的任务并重评各模型；未找到其修订清单。单卡组合配方的容量不足，当前也缺少隔离容器。0 个任务完成。[公开评测器](https://github.com/scaleapi/SWE-bench_Pro-os) |
| DeepSWE 1.1 | 42.2 | Claude Code，温度 1.0，top-p 0.95，256K 上下文。官方 v1.1 在干净环境中对代理提交的代码单独验收。单卡容量与隔离环境均有缺口。0/113 个任务完成。[v1.1 协议](https://deepswe.datacurve.ai/blog/deepswe-v1-1) |
| Agents’ Last Exam | Pass@1 20.4，Score 42.9 | Qwen 未列出 release 和代理配置。完整评测需要匹配的桌面环境和任务资产。当前 v1.1 有 122 个任务，含 3 个 Windows GPU 任务；本地 Docker 子集有 78 个。主机缺少可访问的 KVM 与就绪的虚拟机，组合服务也以纯文本模式启动。0 个任务完成。[官方发布与支持矩阵](https://github.com/rdi-berkeley/agents-last-exam) |
| IFBench | 79.5 | 300/300 题公开协议复测完成，每题采样一次。题目 loose **39.33%**，strict **34.33%**。实际输出及评分存入[本次结果目录](../results/paper-eval-20261119)。 |

SWE 两项的 256K 是 agent 的有效上下文窗口。向组合服务申请 262,144-token 池时，SGLang 将其调整为 138,747，调度器报告单请求输入上界参数为 138,741。另一次测试发送 262,143 个输入 ID 时，服务先因推测解码的额外 token 超过架构长度而返回 HTTP 400；这条请求不能用来测量实际池边界。Claude Code 已安装。容器、桌面和上下文是独立的执行条件。[运行时容量](../results/paper-eval-20261119/final-256k-server.json) · [主机检查](../results/paper-eval-20261119/agent-host.json)

## 实际运行的优化组合

服务使用本地 NVFP4 目标模型与 NVFP4 embedding、FP8 E4M3 KV、BF16 GDN 状态、外置量化 MTP、C1/D4 图内验证、GDN 校验精度与卷积顺序适配、GDN prefill 图和全注意力层 MLP 融合。固定单并发，IFBench 用 48,128-token 池，chunked prefill 为 1,024。NGRAM/DFlash 属于其他推测后端，关闭 radix 的容量配方也单列。首次将 NVFP4 embedding 和 MTP 合并时，旧的共享逻辑在加载阶段报错。修复后草稿模型共享完整的目标 embedding 模块，包括量化尺度与查找表。日志记录了模型加载、图捕获与生成。[启动脚本](../scripts/serve-combined.sh) · [集成补丁](../patches/0009-qwen-packed-mtp-sharing.patch) · [原始报错](../results/paper-eval-20261119/combo-before.log) · [正式运行参数](../results/paper-eval-20261119/ifbench-server.json)

SGLang 基线提交为 `c3c292b65ab42efd933e0c77cfb48eb69f1b81d0`，叠加本仓库改动。目标模型父权重 revision 为 `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33`，草稿父权重 revision 为 `319f741cce68d7914884900c138a1fbb70a42f30`。本轮测量优化部署的绝对质量。量化、推测解码、采样与输出预算各自造成的变化，需要同题同协议的对照才能归因。

## IFBench 公开协议

采用 27B 发布期间可用的 Ai2 数据与评测代码，固定提交 `270c25d8bf1bd6bfb10e69fdc4990432fc61f983`，使用该提交的 `uv.lock`。相较后续版本，当时的数据有两行发生过修改。数据集含 300 个不同的 prompt/key，单条原始 user 消息按 Qwen 自带 chat template 发送。保留默认 thinking，仅取解析后的 final content 评测。全部题目均进入分母，达到上限且没有 final content 的题目按空答案交给官方检查器。[官方评测代码与数据](https://github.com/allenai/IFBench/tree/270c25d8bf1bd6bfb10e69fdc4990432fc61f983)

采样遵循 Qwen thinking 建议，温度 1.0、top-p 0.95、top-k 20、min-p 0、presence penalty 0、repetition penalty 1。输出预算和种子沿用该版 Ai2 生成客户端默认值，分别为每题 4,096 token（含 reasoning）和 42。官方 Ai2 论文通常使用 greedy 和 prompt-level loose 指标；Qwen 模型卡没有注明其 IFBench 行实际使用的输出上限、种子、采样次数或评分变体。输出预算和单次采样会影响本地数值的可比性。[Qwen 采样说明](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md) · [Ai2 客户端默认值](https://github.com/allenai/IFBench/blob/270c25d8bf1bd6bfb10e69fdc4990432fc61f983/config.py) · [实际请求参数和输入校验和](../results/paper-eval-20261119/ifbench-protocol.json)

在仓库根目录使用以下命令可复跑。第一个命令单独启动服务。生成客户端将每题答案、finish reason、耗时和 token 用量追加到 JSONL，可在原目录继续中断任务。重新开始一次独立采样应使用新结果目录。

```bash
PORT=30019 bash scripts/serve-combined.sh
```

```bash
benchdir=.cache/paper-eval/IFBench-release
out=results/paper-eval-20261119
# 准备固定版本；已有同一提交的 checkout 可以直接使用。
git clone https://github.com/allenai/IFBench.git "$benchdir"
git -C "$benchdir" checkout 270c25d8bf1bd6bfb10e69fdc4990432fc61f983
(cd "$benchdir" && uv sync --frozen --no-dev --no-install-project)
.venv/bin/python bench/paper_ifbench.py "$benchdir/data/IFBench_test.jsonl" "$out"
"$benchdir/.venv/bin/python" bench/score_paper_ifbench.py "$benchdir" "$out"
```

## IFBench 全量实测

| 口径 | 通过数 / 总数 | 准确率 |
|---|---:|---:|
| 题目 loose | 118 / 300 | 39.33% |
| 题目 strict | 103 / 300 | 34.33% |
| 指令 loose | 129 / 344 | 37.50% |
| 指令 strict | 114 / 344 | 33.14% |

单卡累积请求耗时 2.22 小时，单题中位数 32.3 秒。自然停止 131 题，触及 4,096-token 上限 169 题，最终答案为空 162 题。输出共 920,430 token，其中推理 898,215。空答案和截断均已计入上表。[分数与分母](../results/paper-eval-20261119/ifbench-summary.json) · [逐题答案及耗时](../results/paper-eval-20261119/ifbench-responses.jsonl) · [逐题评分](../results/paper-eval-20261119/ifbench-scores)

**11 月 21 日 16K 输出预算截断归因抽样（`results/ifbench-trunc-diag-20261121/summary.json`）：** 从原 169 道触及 4,096 上限的题目中按 `seed=42` 抽取样本、将 `max_tokens` 放宽至 16,384（同模型、同采样、同官方评分器），1,800 秒内完成 18/20 题：原 4,096 下 18 题 **0/18（0.0%）** 通过，放宽至 16,384 后立即恢复至 **10/18（55.56% strict & loose）**；其中 13 题自然停止（自然停止通过率 **10/13 = 76.92%**），仍有 5 题触及 16,384 上限且答案为空，样本均值/中位输出长度达 9,685 / 10,727 token。按自然停止组（113/131）叠加截断组恢复率推算，全量 300 题在 16K 预算下约 **~69.0% loose / ~68.0% strict**，证实原 39.33% 低分主因是 4,096 输出上限截断了 thinking 过程，而非量化或 MTP 质量退化。

| 结束原因 | 题数 | 题目 loose 通过 | 题目 strict 通过 |
|---|---:|---:|---:|
| 自然停止 | 131 | 113 | 98 |
| 达到输出上限 | 169 | 5 | 5 |
