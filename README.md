# ICCAD 2026 Problem C

FloorSet研究工作区。先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。

当前最佳整合为[E25基线](experiments/E25_integrated_baseline/README.md)：冻结E17 best51模型，采用E24 steady原生快速后处理。固定v5开发500例全部硬合法，加权Q **1.039578**、平均求解 **0.259487秒**；两项目标尚未同时达成。研究已按用户要求停止。

[E23容量对照](experiments/E23_capacity_geometry/README.md)保留，正式训练尚未启动。日常实验区只保留E25、E23及[无训练baseline](experiments/baseline/README.md)。其余实验和必要证据归入[本次历史归档](history/20261020_E25_consolidation/README.md)，旧记录不改写。

独立环境`.conda/envs/iccad2026`；实际数据位置由[paths.json](protocol/paths.json)解析。原始划分与各版工作集均冻结，当前训练使用[reference_workset_v5.json](protocol/reference_workset_v5.json)，测试继续封存。数据、环境、缓存及权重留本机，不入Git。

数据处理复用[datasets](docs/datasets.md)，评分复用[scoring](scoring/README.md)，布局图统一使用[visualization](visualization/README.md)。上述Q是MIB派生任务的开发中性时间指标，不是正式比赛分数。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [质量目标](docs/aim.md) · [题目全文](references/official/problem_statement.md)
