# E158 三层 attention 初稿生成器

2026-11-20 用户将 E158 收缩为仅研究 RoleInitial，并追加 L0 使用原重叠监督的0.25倍。旧 grouping/夹挡与全链设计搁置。本实验仅获准备授权，正式训练保持关闭。

保持原每模块16维角色特征（含多跳/锚点）、17维公开特征、完整引用端口、MIB 与静态关系bias；width64、4头。RoleInitial 由两个扩展为三个独立参数的 RoleBlock。每块依次进行 attention/公开图局部聚合与残差、原64→192→64 FFN与残差，再把完整隐藏特征传给下一块。只在第三块之后使用一次原64→64→3 head生成 L0。没有内部布局解码、几何反馈、S0/S1/S2或后处理。

L0 唯一输出权重1。总Loss为 1.0R + 0.25A*O + 原wire/compact + 原grouping/boundary + 原公开F/P尺寸和P位置 + 2B*MIB。A=0.08975162861857286，O完整包含penetration、module mean load、module max excess10，故有效重叠系数为0.022437907154643215；不是只降低其中某一项。grouping及其系数保持原件，不加入新瓶颈/夹挡项。所有F/P渠道独立预测，规格只作公开条件与监督。

训练、固定train500/dev500、唯一best、LR和早停使用同一含reference及0.25重叠的总Loss。另记录原无重叠L0口径及完整重叠口径用于诊断，不用于重新选模。E155旧Loss与新总值不能直接横比；本轮同时改变深度和重叠强度，不作单因素因果断言。

复用新random_mib40k_v1 train36k和固定监测面板及持久化CPU缓存，不重抽样。scratch seed0，batch64/尾批32，原轴互换增强、FP32、AdamW与数值门禁/参数投影、四档LR及早停不变；累计4h/200轮上限。只训练本初稿，冻结own-best后停止，无Q、后处理、后段或test。

准备入口：`python -B -m experiments.E158_role_initial3.preflight`。仅使用登记5个train例，包含最大数据padding、三块attention/FFN和唯一head的实际调用、Loss拆项及重叠梯度、公开输入/标签隔离、排列与轴交换、CUDA64/尾批32、无更新捕获和恢复下一更新核对。准备更新全部可丢弃，不复用到正式scratch。

正式启动入口已准备：`python -u -B -m experiments.E158_role_initial3.run_queue --start`。authorization.json 未激活时命令必须拒绝启动。正式训练未启动、无启动时间或训练成绩。
