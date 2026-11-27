# 候选5允许关系集合监督启动

C4完整训练/控制结果已核对并报告，见candidate4_result_report.md。C5保持相同80517参数关系模型、冻结C1物理草稿和相同3形状步+1投影求解路径、相同初始权重及所有Loss系数，仅把唯一原生轴监督改为允许轴集合监督。新目标独立登记candidate5_plan.json，不是C4普通恢复，也不是dev系数扫描。

teacher原始保存中心与nominal/locked WH用于标签：自由形状按公开1%面积下界sqrt(.99)构造最小半尺寸和，固定形状用精确WH；所有满足局部分离下界的轴都允许，负log概率对允许轴概率求和。双轴均允许时，任意轴概率分配都不罚；空集合不丢pair，回退teacher原生max轴。对每个允许轴监督signed rank最小顺序margin，rank teacher-center仍以.25锚定。此集合是局部容许条件，不保证整体锚点可行；现有dev审计4031空集pair明确保留，不将其称teacher原解非法。

预检原五train33.65秒：真实Dropout/Adam eager/graph更新参数与动量差0；重新建图恢复下一步模型/Adam/RNG逐位一致，捕获无额外更新、显式非有限teacher输入门禁/L与C1 F冻结通过。另验证双轴均允许的两个单位方块不因任意轴概率被罚，而唯一水平允许却预测竖向会受罚。

初始模型逐张量严格等于C4已评价的untrained.pt，故无学习对照500例直接复用runs/20261102_candidate4_relations_seed0/control。该控制Q1.237615、12行回退、500合法，完整均时.12966秒/P95 .21073秒/最大.45742秒，一个超过.25秒；不重复计时挑选更快结果。如果C5唯一own-Loss best仍初始，则终末也只读复用相同已完成求解。

后台命令/PID见candidate5_launch.json，日志candidate5.stdout.log/stderr.log。训练器实际登记5/8，1小时/100轮、原全局UTC2026-11-02T02:58:12截止不变。fresh AdamW seed0、batch32 FP32/no AMP/TF32、LR .001/.00025/.0001/.00005；同一固定Loss用于反传/固定train/dev500/best/LR/早停；best/latest保留完整Adam/RNG/order/cursor/seen/调度/累计预算。

candidate5_pipeline仅此候选：唯一own-Loss best一次dev500完整逐例请求含关系传输/边构建/全部几何/硬检查、同100四阶段图；raw2000物理草稿冻结C1，仍复用C1原证据，不重复相同推理。控制只做逐张量等价核验后的引用，不另开求解。测试未读、原split/冻结索引保持。C1-C4绑定源码与当前C5哈希重新核验未变；活动C5及共享关系模型/求解器冻结，改目标/数据/数值另登记。
