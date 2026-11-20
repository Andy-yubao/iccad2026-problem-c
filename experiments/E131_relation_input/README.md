# E131 关系输入64×4（已停止）

两组full/pure共用400→128→64编码、64×4/4头/SwiGLU128及128→64→3读出；full保留关系偏置和连接/G聚合，pure块仅接收h/valid。参数268387/235331。固定16k、FP32/batch32、原面积/无MIB/全模块预测，公开地址按登记随机流重命名，不保证任意重命名不变性。

2026-11-12用户先授权启动，后明确停止并改做独立E13296×6。当前E131状态为user_stop：full完成93轮/46929接受步，pure未启动；完整model/Adam/RNG/排列/游标/调度及累计2108.6928秒保留。STOP/QUEUE_STOP和禁用授权保持，不能因旧启动授权恢复。

配置见 `plan.json`，最新授权/停止记录见 `authorization.json`、`queue_state.json`，训练证据在runs与launch_logs。此前“仅准备”与“已启动”是历史阶段。本实验没有后处理/Q/test；E132是全新scratch，未继承本组权重。
