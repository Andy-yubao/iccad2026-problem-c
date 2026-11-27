# 第二窗口候选3：联合扩大全局几何修正网络

严格保留原C3 own-Loss best的E74和小F全部读出；小F后添加4个width192/heads4归一化RectangleBlock适配块，原小F241547参数与新适配器1974355参数联合训练，共2215902参数。新适配器输入/四层初始化于冻结E74，逐层归一化、fresh归一化读出，仅新增最终readout清零，初始函数逐位等于C3。没有加载旧C6负饱和权重或C7训练适配器。模型类必须使用round2_capacity.DeploymentModel，不能误用normalized_geometry（后者会冻结draft梯度）。

固定Loss、3.6876450434943204权重、训练16k/监测train500/dev500、warm start C3、fresh AdamW seed0、batch32 FP32/noAMP/TF32、LR .00025/.0001/.00005/.00002均同R2C2。没有重新标定或扫描系数；目标仍为C3基础+公开边界/当前几何MST接触任务。架构与训练参数集扩大是新候选，增加随机初始化也会改变Dropout消耗，不能当纯参数量单因素归因。部署端仅额外GPU网络计算，端口保持模块摘要，完整E85单线程后处理不变。

35.81秒原五train预检通过：eager/graph真实Dropout/Adam参数及动量差0，重建恢复下一步model/Adam/RNG逐位一致，非有限teacher在Adam前拒绝、L逐位冻结、源C3各tensor严格保留，初始输出严格相同、NN仅LEGACY输入。第二次更新后适配器pre-SiLU RMS .57–.62，正比例48.5%–55.0%，qkv梯度范数.000578，原小F最大参数梯度范数.5764，未见旧C6的全负饱和。5组公开合成成本MST与CPU精确权重/边数相同。

已启动唯一隐藏后台pipeline，见round2_candidate3_launch.json/stdout/stderr。实际训练才计本窗口第3槽；最多3600秒/100轮及原09:04:02 UTC截止，完整best/latest优化器/RNG/order/cursor/调度/累计预算保留。结束只评价自身Loss唯一best一次全流程dev500、新raw2000、同100图。若epoch0完整C3源未变且adapter零输出，剪去恒等适配器并复用C3已完成控制/raw/图，不重复求解。测试封存，所有数学库/求解器1线程、单GPU串行。
