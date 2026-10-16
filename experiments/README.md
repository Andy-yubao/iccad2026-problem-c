# 实验索引

当前只保留一套基线；代码、说明、运行结果与问题诊断集中维护。

| 实验 | 状态与证据 | 入口 |
|---|---|---|
| 无训练硬约束基线 | 固定 dev5 四组 20/20 硬合法，search 中性时间质量分 1.9090；保留原版 SA 失败输出和五例诊断作为对照 | [说明](baseline/README.md)、[实现](baseline/optimizer.py)、[结果](baseline/runs/stage1_dev_run1/results.json) |

仅复用 datasets 数据处理与 scoring 官方评分。后续新增实验必须先明确假设、对照、面板、预算和可证伪判据，遵守 [执行协议](../docs/protocol.md) 与 [结构规范](../docs/workspace_structure.md)。

已废弃方向按用户要求从工作区与旧源码归档中删除，不作为后续默认架构、参考权重或续跑起点。原始数据、官方资料、现有基线证据与独立环境保留。
