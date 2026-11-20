# E156：冻结 L0 后联合训练两次共享 S0

2026-11-19用户批准完整方案并授权准备后后台启动。冻结E155 best51（父权重仅用于L0）；全新增强题目编码器和一个双层S0从头训练，S0共享调用两次，输出L1至L4，跨调用仅传布局。全部模块包括F/P仍独立预测。

损失：L2和L4各使用reference0.75、原公开任务系数、MIB系数2B及grouping瓶颈最终系数0.2，无绝对overlap或L0移动惩罚。总Loss=0.2*L2_base+0.8*L4_base+A*relu(O4-stopgrad(O0))，其中A=0.08975162861857286，O0取未扰动的冻结L0。相对重叠项在层加权之外只加一次。

训练输入每例75%原始L0、12.5%位置均匀扰动（每轴±0.1sqrt(area)）、12.5%交换一对随机起点及最近邻的中心；只改S0布局输入，不改面积/规格/约束或监督。监测和选模固定使用干净L0输出，关闭dropout与扰动。L0始终eval、无梯度，不迁移历史后段或Adam。

复用新random_mib40k_v1 train36k/固定train500/dev500，batch64尾批32，seed0，原四档LR和平台早停、4h/200轮上限。每轮另记录L0/L2/L4的损失分项、重叠、MST同组边被异组矩形阻隔的数量、完整换侧及有益换侧；诊断不参与梯度、选模或Q。

本轮仅S0，结束冻结own-Loss-best为frozen_s0.json，不训练S1/S2、不后处理、不读test。日志background/queue.session.log，心跳runs/e156_s0_backend_seed0/heartbeat.json。
