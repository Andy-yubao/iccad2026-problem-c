# E119 public-priority v1 — 2026-11-12

用户授权新 Loss 运行 E119，准备后独立隐藏后台启动，正常训练结束自动原固定 dev500 后处理。
本次单组 scratch seed0，原 E119 两初始全局块＋12几何块/共享头/712275参数；36k、原 train500/dev500、FP32 batch32、轴交换/Dropout/原AdamW/LR/门禁、总4h或200轮。
唯一改变：每层原 Loss 新增独立按约束子集平均的公开固定尺寸及预放置中心项，各系数1；总监督仍 .2L4+.8L12，原 reference/G/B/碰撞/线长系数不动。全部模块仍预测，草稿不覆盖公开规格。
反传、监测、best、LR、早停统一新 Loss；original_final_total 只作诊断。完整latest保留Adam/RNG/排列/游标/调度/预算，best独立保存。
复用完整兼容 CPU 缓存，禁止静态预处理重建；同原五训练例仅一次可丢弃受保护 CUDA 更新，正式训练重新scratch。
独立控制器训练→唯一own-new-Loss-best冻结→原dev500/原E98.depth_budget.v1后处理/无MIB本地Q/独立硬检查/完整时间。无test、无新增图、无额外候选或seed，错误/用户停止不推进、不自动重试。
状态queue_state.json，训练runs/global2_seed0/heartbeat.json，日志launch_logs。仅本变体STOP/QUEUE_STOP或其run/evaluation STOP控制本次新授权；所有历史STOP和结果保持。
