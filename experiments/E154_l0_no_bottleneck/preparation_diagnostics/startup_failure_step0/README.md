# E154：移除瓶颈项后从头重训 L0

2026-11-19 用户授权只启动 L0。复用 E153 原32k/train500/dev500缓存、模型及优化器设置，seed0从头初始化，无权重或Adam迁移。L0去掉grouping瓶颈项，其余损失不变，MIB保留。最多4小时/200轮，原四档LR与平台早停。完成后冻结own-Loss-best并停止，不启动后段或后处理。

后续方案登记：仅S0末端L4使用grouping瓶颈项，绝对系数0.2；L0/L8/L12为0。未改全局B及原grouping、MIB系数。后段未授权启动。

日志：background/queue.stdout.log；进度：runs/e154_initial_seed0/progress.json；心跳：同目录heartbeat.json。
