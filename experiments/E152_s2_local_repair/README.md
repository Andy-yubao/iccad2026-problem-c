# E152 只微调 S2 的 A/B 对照

2026-11-18 按用户要求作废 v1 的全部结果并重新开始。两组仍从同一 E151 best45 初始化，冻结到 L8，只更新共享 S2；保留两次调用。A 保留 L12 官方参考监督，B 去掉。两组的 L8 改动惩罚系数均为 0.2。

使用原 36k 训练集、原 train500/dev500，batch64（每轮末批32），每组最多 10 个 epoch，即最多 5630 次更新。学习率按接受更新步从 1e-4 余弦降至 5e-5。每组从 E151 best45 和空 AdamW 重新开始；旧 E152 的权重、优化器、进度和最佳点全部不复用。仍保留每组两小时保护上限，异常或 STOP 阻断后续。

先训练 A 和 B，再按相同后处理预算评价两组各自的 dev Loss-best 和原 E151 best45。实时状态见 `queue_state.json`，逐轮监测和检查点在 `runs/`。旧版现场已移至 `history/E152_discarded_v1_20261118/` 并标为作废，不纳入本轮比较。

准备检查通过后的启动入口为 `python -u -B -m experiments.E152_s2_local_repair.run_queue --start`。终末汇总为 `comparison.md`。见 [冻结定义](DESIGN.md)。
