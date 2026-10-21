# ICCAD 2026 Problem C

FloorSet研究工作区。先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。

当前研究转向[最小依赖修复器的草稿生成](experiments/E29_minimal_repair_draft/README.md)：已定义原始高合法率、固定小幅排斥修复及质量/速度联合目标；先完成目标与机制筛选，再决定新模型，不自动启动长任务。E25继续作为冻结工程基线。

[E28学习式迭代草稿](experiments/E28_iterative_draft/README.md)主候选已完成：51轮早停/best48，原始dev500及完整dev2000均零硬合法、Q=10。重叠减轻但仍未跨越可行性，未替换E25；其他三组不自动运行，测试继续封存。

当前最佳整合为[E25基线](experiments/E25_integrated_baseline/README.md)：冻结E17 best51模型，采用E24 steady原生快速后处理。固定v5开发500例全部硬合法，加权Q **1.039578**、平均求解 **0.259487秒**；两项目标尚未同时达成。[E26尾部研究](experiments/E26_tail_constraints/README.md)已按用户要求暂停，后续Loss与容量方向仅记录，E25继续冻结。

[E23容量对照](experiments/E23_capacity_geometry/README.md)已完成；扩大参数改善了训练拟合，但没有明确开发/最终Q收益，当前最佳继续保留E25。[E27重叠权重对照](experiments/E27_overlap_weight/README.md)已完成best69固定500例验证，全部硬合法；加权Q1.061580，草稿重叠减轻但最终加权Q未优于同版E17，保留E25。训练由用户停止于75轮，未续训。保持50万参数与参考监督，仅将训练重叠系数0.5提高至1.0。E23、暂停的E26研究及[无训练baseline](experiments/baseline/README.md)继续保留。其余实验和必要证据归入[本次历史归档](history/20261020_E25_consolidation/README.md)，旧记录不改写。

独立环境`.conda/envs/iccad2026`；实际数据位置由[paths.json](protocol/paths.json)解析。原始划分与各版工作集均冻结，当前训练使用[reference_workset_v5.json](protocol/reference_workset_v5.json)，测试继续封存。数据、环境、缓存及权重留本机，不入Git。

数据处理复用[datasets](docs/datasets.md)，评分复用[scoring](scoring/README.md)，布局图统一使用[visualization](visualization/README.md)。上述Q是MIB派生任务的开发中性时间指标，不是正式比赛分数。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [质量目标](docs/aim.md) · [题目全文](references/official/problem_statement.md)
