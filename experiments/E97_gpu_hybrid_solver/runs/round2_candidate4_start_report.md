# 第二窗口候选4：释放同组教师拓扑，保留物理分离

严格从原C3 own-Loss best暖启动student.DeploymentModel，241547个小F参数及训练读出保留、E74冻结，无新增层、关系头或CPU步骤。完整原E85单线程后处理不变。

固定Loss沿用R2C2公开边界+当前几何MST任务及系数3.6876450434943204，不重新标定、不扫描。基础项保留teacher中心1/shape.25；near-relative.25只用于不同组/无组近邻，按保留pair数量归一化；同组pair不再模仿教师相对位置或指定轴/方向，而使用实际解码后abs中心差/半尺寸和的最大轴分离程度，relu(1-max_axis)^2惩罚穿透。组内全部自由pair参与；与非组近邻原teacher-axis clearance合并取每例最坏。中心/shape意图和组外分离仍保留。反传/train500/dev500/best/LR/早停用同一无权逐例总Loss。

固定train500梯度诊断无优化器/solver/dev/Q：relative与接触梯度反向21.8%/500例，clearance与接触反向49.16%/238活跃例，其余262零；中位余弦+.1416/+.000334，不能称普遍冲突或证明退化因果。合成反例更直接：教师水平相邻、预测竖直合法接触，新group-relative/clearance/public-contact均0；真实穿透保持正罚；非同组仍保留旧teacher轴要求。

34.21秒原五train预检通过，真实Dropout/Adam eager-vs图更新参数及动量差0，重建恢复下一步model/Adam/RNG逐位一致，非有限teacher在Adam前拒绝、L逐位冻结、源C3严格保留、NN输入仅LEGACY。上述语义反例与既有MST/边界实施检查通过。

seed0 fresh AdamW/batch32 FP32/noAMP/TF32/CPU1，LR .00025/.0001/.00005/.00002，最多3600秒/100轮且原09:04:02 UTC截止。完整best/latest优化器/RNG/order/cursor/seen/调度/累计预算保留。唯一后台pipeline已启动，实际进入训练计第4槽；自身Loss唯一best一次完整dev500、新raw2000、同100图。epoch0整个模型严格等C3则复用现有控制/raw/图。测试封存。

其他方向已登记诊断后停下：来源选择保存几何反事实Q1.010831仍未到目标，不训练selector；固定GPU刚性移动原五train0个accepted、两soft例未改善，其中一例额外.21548秒，没有进入dev或追加强CPU求解。原五例不能证明全库无效，但不足以继续正式投入。所有旧候选证据保护。
