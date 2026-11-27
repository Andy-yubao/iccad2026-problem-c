# 第二窗口最后候选6：L最后四层与F联合训练

从R2C5自身加权Loss唯一best严格加载，SHA028ea3aa124c8cd9e7c3cc7bf8587c699ab9b4932f256899e3780d4d8c8816e0，所有读出保留。模型类round2_joint_placer.DeploymentModel，部署结构与student相同；训练时重新从LEGACY公开输入计算L，再经过F，不再将缓存teacher base当网络草稿输入。教师target只监督，不进入NN。

这是明确登记的L/F新阶段：训练L的layers.8–11及F，2170779参数；L前8块、input/state_embed/共享head等全部冻结并eval。F基础LR .0001/.00005/.00002/.00001，L最后4块固定0.1倍，各组倍率贯穿调度。fresh AdamW seed0/batch32 FP32/noAMP/TF32，Loss/公开N权重/3.687645公共项系数逐字同C5，同函数反传/监测/best/LR/早停，无dev扫系数。部署额外层和CPU步骤均为0，完整旧E85单线程后处理不变。

35.83秒原五train预检：eager/graph实际两参数组Dropout/Adam更新与动量差0，重建恢复下一步model/Adam/RNG逐位一致，非有限teacher在两组Adam前拒绝；源C5初始函数逐位相同，L-last4实际改变，其他L逐位不变，L最大参数梯度.40619，峰值allocated2212.86MiB。公开N权重/子项、几何语义反例及NN标签隔离通过。

预检后用户继续时，墙钟已到08:39UTC。为遵守原09:04:02UTC总截止，正式训练尚未启动前，将1200秒上限收紧为600秒、最多40轮，其余数值路径不变。旧1200秒源码/计划归档于原preflight/budget1200_source_before_start.zip；仅预算迁移不重复数值测试，preflight_budget600保存新绑定及明确复用来源。没有清零或延长总窗口。

已后台启动唯一pipeline，见round2_candidate6_launch.json/stdout/stderr。自身Loss唯一best终末一次完整dev500、新L和F的raw2000、同100图；展示L须当前权重实际重算。若best epoch0且整个模型严格等C5，复用C5现有终末/raw/图，不能错误复用C3。所有CPU数学库/求解器1线程，单GPU串行。此候选完成即收尾，不开候选7/8，截止优先；测试封存。
