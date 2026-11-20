# E157 顺序监督与冻结边界

用户明确取消S1/S2联合训练。阶段S1仅训练s1_encoder/s1，冻结initial_generator/s0_encoder/s0。阶段S2仅训练s2_encoder/s2，冻结全部前缀包括s1_encoder/s1；阶段2父权重只能来自阶段1固定dev own-best。两阶段独立新建Adam和RNG种子，不把S1 optimizer迁移给S2。

S1监督L6/L8权重.2/.8；S2监督L10/L12权重.2/.8。完整P包括A*(penetration+module_mean_load+module_max_excess10)、0.5B*(wire+compact)、2B*(grouping_contribution+boundary_contribution)、public F/P size、P center及2B*MIB。R和grouping max-edge均不反传。监测可报告分组瓶颈诊断，禁止将其混入P。

S2仅L12加0.2*(Dcenter+0.25*Dratio)，该项还乘L12层权重0.8。center超0.05sqrt(area)和logratio超0.1才有惩罚；P中心、F/P形变不受移动惩罚限制，仍由公共规格Loss监督。基准冻结L8；每模块L8累计交叠/自身面积>0.01时惩罚再乘0.25，不降低其绝对碰撞Loss。

同一E156 best79来源/源文件和checkpoint哈希在初始化时验证，完整历史前缀推理在短检中逐位核对。S2短检使用可丢弃S1副本检查覆盖、冻结和恢复，正式S1重新scratch。两阶段短检使用登记5个新train样例和全数据padding；GPU64、尾批32、绝对碰撞、无reference标签依赖、无grouping瓶颈、轴互换和序列化下一更新均检查。再用同5例内1题做完整MIB后处理/评分短检，不读dev/test载荷。

源码身份覆盖两阶段全部代码、预算、固定数据清单及父S0描述。动态生成frozen_s1不纳入共同源码哈希，作为S2单独父身份写入配置和checkpoint；避免S1完成后导致自身源码身份变化。新增强数据缓存直接复用，模型独立预测全部F/P位置和形状。
