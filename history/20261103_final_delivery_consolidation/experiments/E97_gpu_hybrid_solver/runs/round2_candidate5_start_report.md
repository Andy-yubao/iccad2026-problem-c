# 第二窗口候选5：固定公开模块数重要性权重

从原C3 unique own-Loss best严格暖启动，保留241547小F全部训练读出、E74冻结。网络/部署结构/完整E85单线程后处理不变，源函数与R2C4初始函数相同。

唯一目标迁移是每例所有基础项和公开任务项乘exp((publicN-120)/12)/0.12578465955104443，然后weighted_base+3.6876450434943204*weighted_public_soft。逐例算术均值的同一加权总Loss用于反传/train500/dev500/best/LR/早停；不按batch归一化、不重采样、不移除小N成员。N来自公开valid mask，常数是固定train16k公开权重平均值，不是reference评分分母。新训练集metadata inventory得到.12578465955103685，与原C8已登记常数差约7.6e-15；无模型、dev、Q、solver或优化器。该公式曾在历史不同C8方案使用，不据此断言本模型会改善。

基础仍是R2C4同组自由轴/方向几何分离、组外教师近邻关系，公开boundary/current MST接触项及其3.687645系数不变、不重新标定。重要性权重对大N提高训练注意，可能损害小N和改变梯度裁剪频率，这些都属于已登记的新Loss。

34.08秒原五train预检通过：实际Dropout/Adam eager-vs图参数/动量差0，重建恢复下一步model/Adam/RNG逐位一致；非有限teacher在Adam前拒绝、L冻结、C3初始函数严格相同、NN仅LEGACY摘要；publicN权重独立数值检查和所有子项加权逐位一致、替代组接触/真实穿透反例及MST实施检查通过。

seed0 fresh AdamW/batch32 FP32/noAMP/TF32/CPU1，LR .00025/.0001/.00005/.00002，最多3600秒/100轮及原09:04:02UTC截止。已启动唯一后台pipeline，实际进入训练才计第5槽。best/latest保存完整Adam/RNG/order/cursor/seen/调度/累计预算。训练结束唯一own-Loss best一次全流程dev500、新raw2000、同100图；epoch0严格等C3则复用已有控制/raw/图，不重复相同求解。测试封存，单GPU串行，无增强CPU求解。
