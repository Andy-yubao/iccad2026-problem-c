# E155：新形状修正版数据从头训练 L0

2026-11-19用户授权准备后后台启动。复用新random_mib40k_v1的36k训练集及其分层随机train500/dev500。公开MIB组面积平均；监督同MIB形状统一、保留原中心。数据已完成，不重建缓存。

模型/Loss/优化器沿用E154：两层width64角色初稿，116043参数；预测所有F/P通道，MIB保留；reference1、无直接overlap、无grouping瓶颈。seed0从头初始化，无历史权重或Adam迁移。batch64，563步/轮（末批32）；LR为0.001→0.00025→0.0001→0.00005，原平台调度与早停，最多4小时/200轮。

仅训练L0，完成后冻结固定新dev500 own-Loss-best并停止；不启动后段、后处理或test评价。日志background/queue.session.log，实时进度runs/e155_initial_seed0/heartbeat.json及progress.json。
