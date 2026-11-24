# E158 正式训练启动记录

2026-11-20 用户明确要求“启动训练吧”。已核对准备检查 passed、源码身份逐项一致、无 STOP/QUEUE_STOP/PAUSE_REQUEST、无既有正式 run 后，激活仅 e158_initial 的正式训练授权并启动隐藏后台队列。

仅三层 attention/FFN 的 RoleInitial；reference1.0、原完整重叠组0.25，其他原Loss不变。scratch seed0、36k、batch64、最多4小时/200轮；结束冻结唯一own-Loss-best并停止，不启动S0/S1/S2、后处理或test。README/DESIGN为准备时已冻结源码身份，原件不改；当前授权和实时状态以 authorization.json、queue_state.json、runs/e158_initial_seed0/heartbeat.json 为准。

启动时间及PID见 background/activation.json 和 background/launch.json；总日志 background/queue.stdout.log，错误日志 background/queue.stderr.log。
