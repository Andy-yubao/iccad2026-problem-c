# ICCAD 2026 Problem C

FloorSet研究工作区。先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。

当前最佳整合为[E25基线](experiments/E25_integrated_baseline/README.md)：冻结E17 best51模型，采用E24 steady原生快速后处理。固定v5开发500例全部硬合法，加权Q **1.039578**、平均求解 **0.259487秒**；两项目标尚未同时达成。模型、数值求解代码及最佳结果保持冻结。

[E29最小依赖修复器的草稿生成](experiments/E29_minimal_repair_draft/README.md)只完成目标定义，未实现或运行新模型；用户要求先同步和整理，暂不尝试研究机制。测试继续封存。

日常实验区保留E25、未运行E29及[无训练baseline](experiments/baseline/README.md)。已结束的E23/E26/E27/E28源码、权重与必要证据归入[20261021归档](history/20261021_E29_consolidation/README.md)，早期整合见[20261020归档](history/20261020_E25_consolidation/README.md)。历史记录保持原样，迁移路径由清单恢复；数据、缓存与权重只保留本机。

独立环境`.conda/envs/iccad2026`；实际数据位置由[paths.json](protocol/paths.json)解析。原始划分与各版工作集均冻结，当前训练使用[reference_workset_v5.json](protocol/reference_workset_v5.json)，测试继续封存。数据、环境、缓存及权重留本机，不入Git。

数据处理复用[datasets](docs/datasets.md)，评分复用[scoring](scoring/README.md)，布局图统一使用[visualization](visualization/README.md)。上述Q是MIB派生任务的开发中性时间指标，不是正式比赛分数。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [质量目标](docs/aim.md) · [题目全文](references/official/problem_statement.md)
