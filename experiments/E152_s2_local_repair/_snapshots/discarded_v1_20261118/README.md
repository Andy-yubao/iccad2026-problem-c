# E152 只微调 S2 的 A/B 对照

按 2026-11-18 用户授权，两组各训练五轮。均从 E151 best45 初始化，冻结到 L8，只更新共享 S2；保留两次调用。A 保留 L12 官方参考监督，B 去掉。两组的 L8 改动惩罚系数均为 0.2。

固定 16k、batch64，每组 1250 次更新。先训练 A 和 B，再按相同后处理预算评价两组各自的 dev Loss-best 和原 E151 best45。实时状态见 `queue_state.json`。两组逐轮监测和检查点在 `runs/`。

准备检查通过后的启动入口为 `python -u -B -m experiments.E152_s2_local_repair.run_queue --start`。终末汇总为 `comparison.md`。见 [冻结定义](DESIGN.md)。
