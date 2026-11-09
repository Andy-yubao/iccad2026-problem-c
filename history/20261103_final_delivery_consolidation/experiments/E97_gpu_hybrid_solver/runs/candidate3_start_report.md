# 候选3启动记录

C2唯一best完整终末证据已完成并检查，见candidate2_result_report.md。C3从C1唯一own-Loss best（5590e672cbb05a862f739452fe6432846287ea906c83782f84a0dfdb5d486b50）暖启动全部小F权重，不清零读出层；保持E74冻结、同241547参数架构、同teacher16500及同slim3，不采用C2收缩后权重。

新目标一次登记为C1 center1 + shape.25 + near-relative.25 + physical-clearance-tail1。间距阈值1来自矩形接触条件：沿teacher分离轴的signed中心距至少为当前两半尺寸之和；每例teacher近邻对平方hinge取最坏值。不含C2轴优势比较项，无dev系数扫描。反传、固定train/dev500监测、best、LR和早停使用同一函数。

启动门槛一：原五train只读teacher nominal重构→same slim3，5/5合法且0行回退；仅五例机制，不宣称全局能力。所有nominal teacher草稿本身均未严格合法，保存了面积容差解码差异。

启动门槛二：原五train真实Dropout/Adam普通/图更新参数差0、动量差0，捕获无额外更新，重新建图恢复下一步模型/Adam/RNG逐位一致；非有限拒绝与L冻结通过。实施33.82秒。另验证两单位方块接触时clearance=0，中心距收缩为.5时clearance=.25，确认排除了C2设计允许的重叠反例。

后台启动命令/PID见candidate3_launch.json，日志candidate3.stdout.log/stderr.log。实际正式计数由训练器登记为3/8，不因预检/唤醒另增。fresh AdamW/seed0、batch32/FP32、LR .001/.00025/.0001/.00005，其余停止与保存见candidate3_plan.json。原UTC2026-11-02T02:58:12全局截止不变，单候选1小时/100轮上限。

candidate3_pipeline仅本候选：自身固定Loss唯一best一次完整dev500、rawdev2000、同100四阶段图；不按Q另选checkpoint，不另起重复评价或候选4。C1/C2绑定源码哈希已核验未变，teacher只读，测试封存。活动C3源码/计划/模型/数据依赖冻结。
