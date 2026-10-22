# ICCAD 2026 Problem C

2026-10-22新一轮[E32学习关系与求解反馈](experiments/E32_learned_relation_feedback/README.md)进行中：用户授权更大规模探索和12小时上限，目标仍为加权Q、完整耗时与尾部稳定性。CPU总计最多两线程，测试封存；E25保留为接受基线。

2026-10-22已完成[E30两线程研究](experiments/E30_stable_fast_geometry/runs/20261022_research_summary/analysis.md)：普通几何微调在固定监测面板改善Q与均时，但三种子及一次额外500例确认没有支持尾部稳定升级，**保留E25**。额外确认Q1.051866→1.046393、均时0.247792→0.241181秒，最差5%均值与最大Q恶化。独立[E31工作区复用检查](experiments/E31_conic_workspace_reuse/README.md)真实流程零命中，五例后停止。该轮数值任务已结束，未耗尽8小时上限；CPU总计最多两线程，测试封存。

FloorSet研究工作区。先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。

当前最佳整合为[E25基线](experiments/E25_integrated_baseline/README.md)：冻结E17 best51模型，采用E24 steady原生快速后处理。固定v5开发500例全部硬合法，加权Q **1.039578**、平均求解 **0.259487秒**；两项目标尚未同时达成。模型、数值求解代码及最佳结果保持冻结。

日常实验区保留E25及[无训练baseline](experiments/baseline/README.md)，当前E32/E33/E34方向见[实验索引](experiments/README.md)，E30/E31保留上一轮必要证据。原E29方案已按用户要求删除，编号29可供后续新实验复用；测试继续封存。已结束的E23/E26/E27/E28源码、权重与必要证据归入[20261021归档](history/20261021_E29_consolidation/README.md)，早期整合见[20261020归档](history/20261020_E25_consolidation/README.md)。历史记录保持原样，迁移路径由清单恢复；数据、缓存与权重只保留本机。

独立环境`.conda/envs/iccad2026`；实际数据位置由[paths.json](protocol/paths.json)解析。原始划分与各版工作集均冻结，当前训练使用[reference_workset_v5.json](protocol/reference_workset_v5.json)，测试继续封存。数据、环境、缓存及权重留本机，不入Git。

数据处理复用[datasets](docs/datasets.md)，评分复用[scoring](scoring/README.md)，布局图统一使用[visualization](visualization/README.md)。上述Q是MIB派生任务的开发中性时间指标，不是正式比赛分数。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [质量目标](docs/aim.md) · [题目全文](references/official/problem_statement.md)
