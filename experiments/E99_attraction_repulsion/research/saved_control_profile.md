# 冻结C3既有500例开销诊断

Q=1.013366283，完整均时/P95/最大={'mean': 0.2108070260004024, 'p95': 0.5261268749942246, 'max': 2.313262299998314}，超250ms=135/500。

NN草稿阶段={'mean': 0.025214398200187133, 'p95': 0.02968399000310455, 'max': 0.0322706000006292}；后处理阶段={'mean': 0.18397498860018094, 'p95': 0.4952908299994305, 'max': 2.2862554999883287}。

逐调用阶段：

| 阶段 | 调用数 | 每例调用均数 | 单次均时ms | 单次P95 ms |
|---|---:|---:|---:|---:|
| shape | 2611 | 5.222 | 24.885 | 46.563 |
| project | 500 | 1.000 | 7.821 | 13.686 |
| certified_rigid_moves | 500 | 1.000 | 0.837 | 3.688 |
| early_projection | 10 | 0.020 | 10.056 | 16.350 |
| full_packed_refine | 7 | 0.014 | 762.935 | 806.602 |

按公开N权重汇总的软违例数量={'boundary_violations': 0.16093350995352146, 'grouping_violations': 0.1581855944975588, 'mib_violations': 0.0}。

Clarabel reported native_seconds can overlap setup accounting; do not add diagnostic timers as disjoint phases. These are old measured rows, not a fresh latency claim.

后续优先：固定全局组织的快速并行前置、减少可行性失败/复杂修复重复；新的显式装配/初始化/solve墙钟分开计时。不能用旧阶段计时直接宣称新工程提速。
