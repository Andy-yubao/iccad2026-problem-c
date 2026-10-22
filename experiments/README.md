# 当前实验

| 实验 | 状态 |
|---|---|
| [E34_sparse_contact_feedback](E34_sparse_contact_feedback/README.md) | 已结束：五例回调加速2.87倍，终末500全部硬合法；Q/时间/尾部联合要求未通过，不扩确认 |
| [E33_learned_solver_portfolio](E33_learned_solver_portfolio/README.md) | 进行中：train1000先导通过互补空间条件，正在补齐train16000四专家矩阵，随后门控训练及内层校准；外层评价延后。共享12小时/两线程预算 |
| [E32_learned_relation_feedback](E32_learned_relation_feedback/README.md) | 两组训练及各自终末500已结束：结构代理改善，但无学习策略通过联合要求；不扩种子/确认，保留E25 |
| [E31_conic_workspace_reuse](E31_conic_workspace_reuse/README.md) | 已结束：SDK复用有效，但五例实际主几何0/33命中缓存；未扩大评价 |
| [E30_stable_fast_geometry](E30_stable_fast_geometry/README.md) | 已结束，保留E25：四次微调、后处理对照、一次额外配对确认；平均指标改善但尾部稳定性未通过，两线程，测试封存 |
| [E25_integrated_baseline](E25_integrated_baseline/README.md) | 当前最佳：E17 best51＋E24 steady，500/500硬合法、加权Q1.039578、平均0.259487秒；冻结保留 |
| [baseline](baseline/README.md) | 无训练硬约束参照 |

E23/E26/E27/E28见[20261021归档](../history/20261021_E29_consolidation/README.md)；此前实验见[20261020归档](../history/20261020_E25_consolidation/README.md)。更早归档保留原目录，不自动恢复历史路线。

原E29方案已按用户要求删除，不保留待实施计划；编号29可供后续新实验复用。
