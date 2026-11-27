# 最后候选8启动记录

C7完整dev500、新raw2000、同100四阶段图与激活记录审阅完成，结果见candidate7_result_report.md。归一化改善训练却未减少11例行回退，因此C8保留相同模型和原生动态slim3后处理，只迁移训练目标，不修改CPU求解器或增加推理步骤。

C8严格暖启动C7自身Loss唯一best（SHA ac003467665ac78dab540f03117c77c86627b5340854e2cfafba642be8504a51），使用normalized_geometry.DeploymentModel，保留全部归一化及head.3已训练读出，冻结原C1 L/F，仅训练1,974,355参数适配器。fresh AdamW/seed0是已登记目标迁移，不冒称普通续训。NN只接收LEGACY完全压缩端口摘要；完整原始端口/连接由Loss读取，不成为NN token。

固定每例Loss：

`exp((N-120)/12)/0.12578465955104443 * (teacher + .5392561010352398 physics + .1320398351856648 corridor + .12665827536621818 quality)`。

teacher沿用C1/C7的center1+shape.25+near-relative.25。physics含精确交面积、归一化穿透、公开边界和兼容MIB；corridor为固定16次原生排序松弛得到的公开锚点路径压力；quality含外框/公开总面积和精确完整连接线长/公开规模归一化。没有reference评分分母。公开模块数权重更强调大实例，采样顺序和成员不改变。

系数来自candidate8_calibration_plan.json预登记的一次train_monitor500局部状态梯度标定，0优化器更新、0solver/dev/Q调用；活跃梯度成员physics500/corridor396/quality500，其余成员仍保留于Loss。预定梯度份额physics1/corridor.5/quality.1，未按dev扫系数。weights.json SHA 031a5cb9bcdcc78da365b8cd39f5c724f5634da4c5c63771802f904c6ac8de2b。局部梯度标定不保证参数梯度平衡、全局可行性或最终Q改善。

预检35.17秒通过：真实Dropout/Adam普通/图参数与动量差0、重建后下一步模型/Adam/RNG逐位一致、非有限输入在Adam前拒绝、L与C1 F冻结、初始权重逐张量等于C7、NN与完整端口隔离。主体梯度非零，读出RMS约.29–.55，峰值allocated1240.25MiB。复用相同结构C7的五例draft延迟证据，不重复计时。另原五train公开输入的独立wire/bbox功能对照通过，线长相对误差最大2.71e-7，0优化器/solver/Q调用；见20261102_candidate8_wire_check/results.json。

后台启动命令/PID见candidate8_launch.json；candidate8.stdout.log已确认train16000/dev500加载与引擎就绪，候选账本已由训练器实际计入8/8。batch32 FP32/noAMP/TF32，AdamW wd.0003/clip1，LR .00025/.0001/.00005/.00002，最多7200秒/200轮及原全局截止。反传、train/dev500监测、best、LR和早停均用同一个固定新Loss。best/latest/untrained保留Adam/RNG/order/cursor/seen/调度/累计预算。

已安排唯一own-Loss best一次原slim3完整逐例dev500、新raw2000及同100四阶段图；若整个epoch0 best逐张量仍等于C7，直接复用C7已完成终末/raw/gallery，不能剪去C7已训练适配器。全部回退计入完整Q，不按Q反选checkpoint、不重复控制求解。正式单例计时含公开构建/传输、NN、全部几何/回退、返回与必要硬检查；加载/预热单列。

这是第8个且最后正式候选，其已登记训练及唯一评价完成后即收尾关闭heartbeat，不开C9。原UTC2026-11-02T02:58:12截止不延长，用户停止与全局期限优先。C1-C8全部成功预检绑定源码已逐项核验无变化，活动源码冻结。teacher不再生成、旧相同评价不重跑、测试封存。teacher容差WH未存、名义面积重构偏差和9个train aspect超5等局限保留；目标未达不得称成功。

启动核实：首轮已完成500次实际Adam更新；固定dev500新Loss从epoch0 .236620910降到epoch1 .232100924，train500从.240521536降到.236387473。仅说明已实际训练，不代表Q改善。stderr为空。
