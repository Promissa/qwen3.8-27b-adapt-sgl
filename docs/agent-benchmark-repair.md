# Qwen3.8-27B 本地代理基准修复与复测

2026-11-19，本轮固定 RTX 5090 D V2（24 GB）单卡、Ryzen 7 9700X，使用 WSL2 Ubuntu 与公开发布的数据。每题一次采样，常规工程题按任务说明逐题选择。选集覆盖编译、查询优化、格式转换、程序修复和中文办公，不能外推整套正确率。原始任务、镜像摘要、参考评分和模型轨迹分别保留。[逐题清单](../results/agent-repair-20261119/selected.json) · [当前评分与覆盖数](../results/agent-repair-20261119/status.json) · [版本锁定](../results/agent-repair-20261119/revisions.json)

| 发布版本 | 本机选集 | 评分方式与协议差异 |
|---|---|---|
| Terminal Bench 2.1，89 题 | pMARS、Cython、C/Python 双语言 Fibonacci、SQLite 查询优化 | Harbor 0.26.0 自带 Terminus 2，官方逐题 verifier。Cython 原版和补固定依赖的复测分开统计。 |
| SWE-bench Pro Public v1.0，731 题 | qutebrowser 的时长解析、命令拼写建议、字体配置 | 公开 v1.0 数据与同期 Scale evaluator，模型在离线容器内修复，再由官方脚本评分。Qwen 修订清单未公开，公开 Pro 结果独立报告。 |
| DeepSWE 1.1，113 题 | ABS 步进切片、TOML 表转换、SQL DDL 格式化 | 在原始基础提交上运行 Claude Code，导出模型提交的差分，由 Pier 在独立环境评分。未提交的编辑按任务规则排除。 |
| Agents’ Last Exam 原始 v1，152 题 | 中文 Basel 事件分类与资本计算 | 使用公开 Ubuntu 桌面、KVM 和 ALE-Claw、本地模型与确定性评分器。Qwen 未披露 ALE 的具体发布版和代理配置。 |

## 服务与评分器验收

- Docker Engine 29.1.3 与 Compose 2.40.3 安装在 WSL 内。任务镜像从其公开注册表取得。Docker 与 containerd 使用当前 WSL 会话的系统代理配置，位于 `/run/systemd/system`，重启后须按本机网络重新配置。官方 QEMU 容器通过创建 KVM 虚拟机检查。[版本](../results/agent-repair-20261119/docker-version.txt) · [KVM 实测](../results/agent-repair-20261119/kvm-create.log)
- 组合服务保留目标、embedding 与 MTP 的 NVFP4、FP8 KV 及原有 GDN/MLP 优化。两项 SWE 增加 `--max-total-tokens 262144 --cpu-offload-gb 4`。调度器报告单请求输入上限 262,138。真实 261,072-token 多轮模板请求在 4 GiB、6 GiB 档均找回首轮标记，各生成 8 token，零回退。端到端分别为 439.07 秒与 166.96 秒，各一次。32-token 短对照两档均与 48K 服务逐 token 一致；这只验证该条提示。[4 GiB](../results/agent-repair-20261119/offload-4.long.json) · [6 GiB](../results/agent-repair-20261119/offload-6.long.json) · [对照](../results/agent-repair-20261119/reference.probe.json)
- Claude Code 2.1.323 的 Messages 请求经 loopback 进入 SGLang。适配器固定温度 1.0、top-p 0.95，交给 Qwen 模板处理默认 xhigh，解决显式 effort 的枚举转换错误。CLI 实际记录 262,144 上下文、32,000 的单次输出上限。本地读文件的工具调用与工具返回已验收。[适配器](../bench/qwen_agent_messages.py) · [CLI 实测](../results/agent-repair-20261119/claude-context.json)
- Pro 的参考补丁和空补丁分别得 1 和 0。DeepSWE 的 ABS 参考提交通过 6/6 修复测试、6/6 回归测试；空补丁分别为 0/6、6/6，二元得分为 0。[Pro 控制组](../results/agent-repair-20261119/swe-gold/eval_results.json) · [Pro 空补丁](../results/agent-repair-20261119/swe-empty/eval_results.json) · [Pier 原始评分](../results/agent-repair-20261119/pier)
- Cython 官方参考解在原环境仅通过 10/11 项，失败来自新发布的 planarity 1.0.0。保留官方基础镜像与 verifier，将 planarity 固定为 0.6 后，官方参考解通过。固定依赖的模型成绩另列。[原始诊断](../results/agent-repair-20261119/cython-debug.log) · [修复后控制组](../results/agent-repair-20261119/harbor/cython-pinned-oracle/result.json)
- ALE 使用含视觉塔的同一套组合。两张本地测试图各实际消耗 120 个图像 token，标签严格正确 1/2，忽略空格为 2/2。Ubuntu 公开镜像逻辑大小为 179,493,797,888 字节，逐段及整盘校验后才进入官方任务。视觉探针与 dry-run 单列。[视觉输出](../results/agent-repair-20261119/vision.json) · [资产实测](../results/agent-repair-20261119/ale-assets.json) · [官方执行清单](../results/agent-repair-20261119/ale-office-dry-run.log)

## 当前模型试题记录

- Terminal Bench 的 pMARS 通过 4/4 检查，得 1，整题约 5 分 14 秒。原版 Cython 达到上游 900 秒代理限时，随后官方 verifier 通过 2/11 项，得 0。[pMARS](../results/agent-repair-20261119/harbor/pmars-model/result.json) · [Cython](../results/agent-repair-20261119/harbor/cython-model/result.json)
- Pro 的时长解析题用 19 轮生成了 3,042 字节补丁，代理耗时 534.25 秒，官方测试通过。其余模型题随逐题状态表更新。[代理统计](../results/agent-repair-20261119/qwen-duration/metadata.json) · [官方评分](../results/agent-repair-20261119/swe-qwen/eval_results.json)
- ALE 的完整磁盘在下载中遇到大文件限速。已从发布方固定版本的磁盘按需读取原始文件系统，取出指定办公题的输入与加密参考归档，逐文件记录 SHA256，并用工作簿内部 CRC 校验四份 XLSX。参考数据仅供评测器在代理交付后使用。原版 QEMU 探针在 900 秒内未等到 CUA，日志显示文件系统检查等待远程磁盘读取。后续可按环境就绪顺序使用发布方 Docker 镜像、完整磁盘或已验收的只读按需磁盘。按需磁盘的可写副本在已确认干净的 ext4 上仅更改启动检查参数。[题目资产](../results/agent-repair-20261119/ale-guest-inputs.json) · [启动检查](../results/agent-repair-20261119/ale-boot-profile.json)

## 查看结果

在项目根目录执行以下命令可从原始评分输出重建状态。每题保留明确状态。环境修正版、参考解和空补丁均独立于原版模型选集。已启动的任务可在评分器结束后重建统计。

```bash
.venv/bin/python bench/agent_repair_status.py results/agent-repair-20261119
```

来源见 [Terminal Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)、[SWE-bench Pro 公开评测器](https://github.com/scaleapi/SWE-bench_Pro-os)、[DeepSWE 1.1](https://deepswe.datacurve.ai/blog/deepswe-v1-1) 和 [ALE 原始 v1 发布说明](https://github.com/rdi-berkeley/agents-last-exam/blob/main/docs/releases/v1.md)。

## 11 月 20 日续跑

WSL 在上一条 ALE 轨迹写入后重启。续跑复用了已经完成的评分，恢复了本机模型接口、离线容器连接及单卡队列。Pro 的时长解析、命令建议两题均通过。DeepSWE 的步进切片提交通过 5/6 修复测试和 5/6 回归测试，按官方规则得 0。ALE 首次环境错误及重启时未完成的轨迹均不计模型分数。

Fibonacci 和 SQLite 原始 Harbor trial 未取得 verifier reward。续跑从对应 Terminus 轨迹中的最后一次完整文件写入还原提交，使用原始 Docker 任务镜像、原版测试文件、Python 3.13、pytest 8.4.1 和 pytest-json-ctrf 0.3.5 离线验收。文件回放单列于逐题 JSON 的 `artifact_replay` 字段。Fibonacci 的原始代理超时，回放通过 1/1 项，覆盖编译、Python 执行和数值核对。SQLite 回放通过 6/6 项，耗时 139.64 秒，覆盖结果等价性、查询耗时及数据库完整性。[Fibonacci 回放](../results/agent-repair-20261119/polyglot-c-py-recovered) · [SQLite 回放](../results/agent-repair-20261119/query-optimize-recovered) · [当前覆盖数](../results/agent-repair-20261119/status.json)

第二次恢复沿用已完成的 ALE 桌面试验。官方评分器报告三份交付文件均缺失，得分为 0，耗时 259.63 秒。轨迹显示多条 shell 工具参数进入了图形动作解析器。另有一次内存摘要请求走到外部 API，收到认证错误。额外复测使用 ALE-Claw 的 Chat Completions 工具转换，接入 SGLang 原生工具解析，并将主模型和辅助推理调用固定到本机端点。原始评分和修正版分数在 `status.json` 中各有记录。[ALE 原始记录](../results/agent-repair-20261119/ale-office.log) · [本机适配器](../bench/ale_qwen_local.py)

DeepSWE 的 TOML 转换题曾因 20 分钟网关超时而中断，当时未导出提交。修复将网关和 CLI 的单请求上限延长至 90 分钟，两项 SWE 仍使用各自的任务时限。WSL 重启时正在初始化的字体和 SQL 格式化任务已重新排队。[超时试验元数据](../results/agent-repair-20261119/qwen-tomlkit-transport-504/metadata.json) · [续跑状态](../results/agent-repair-20261119/status.json)

ALE 原生工具复测运行 326.8 秒并交付全部三份文件。事件类别与损失金额各 60/60，资本计算正确，业务线 50/60（83.33%），低于评分器要求的 90%，整题仍得 0 分。最早的运行没有交付文件。本次改善体现在交付和分项准确率。[原生工具复测日志](../results/agent-repair-20261119/ale-office-native.log)

公开 Pro 子集三道 qutebrowser 题已全部评分，时长解析和命令建议通过，字体配置失败。字体题在 72 分 46 秒内生成 25,920 字节补丁，官方参考补丁得 1，空补丁得 0。官方得分为 2/3。[字体模型评分](../results/agent-repair-20261119/swe-fonts-qwen/eval_results.json) · [参考补丁](../results/agent-repair-20261119/swe-fonts-gold/eval_results.json) · [空补丁](../results/agent-repair-20261119/swe-fonts-empty/eval_results.json)

## 本机选集最终结果

11 道选题均已执行。9 道取得框架评分，2 道 Terminal Bench 题取得文件回放结果。各结果适用于逐题筛选的本机选集，分母见下表。[逐题分数、提交与超时记录](../results/agent-repair-20261119/final-subset.json)

| 基准 | 框架已评分 / 选题 | 满分题数 | 关键结果 |
|---|---:|---:|---|
| Terminal Bench 2.1 | 2/4 | 1 | pMARS 得 1，Cython 得 0。Fibonacci 的代理超时，SQLite 的 verifier 超时，文件回放分别通过 1/1 和 6/6 项。 |
| SWE-bench Pro Public v1.0 | 3/3 | 2 | 时长解析、命令建议通过，字体配置失败。 |
| DeepSWE 1.1 | 3/3 | 0 | 步进切片的修复和回归各通过 5/6。另两题到达官方 3 小时时限，未形成提交。 |
| Agents’ Last Exam 原始 v1 | 1/1 | 0 | 原运行无交付。原生工具复测交付三份文件，业务线 50/60，最终仍为 0。 |

SQL 格式化和 TOML 转换的退出码均为 124，提交差分均为 0 字节。Pier 在独立容器中评分，SQL 格式化的修复测试为 0/32，TOML 转换为 0/60。两次完整推理复测未触发最初的 20 分钟 HTTP 504，均到达题目的 3 小时截点。[SQL 代理元数据](../results/agent-repair-20261119/qwen-sqlfmt/metadata.json) · [TOML 代理元数据](../results/agent-repair-20261119/qwen-tomlkit/metadata.json)

在单卡 256K 配方下，字体配置题的实际请求达到约 186K token。长窗口通过了 261,072-token 合成探针和真实代理多轮请求，代理仍可能在题目时限前来不及提交。接入与容量修复应按工具可用性、输入长度和交付完成率分别验收。[字体运行日志](../results/agent-repair-20261119/fonts-server.log) · [256K 探针](../results/agent-repair-20261119/offload-4.long.json)
