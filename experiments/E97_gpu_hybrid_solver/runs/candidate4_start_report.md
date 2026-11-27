# 候选4关系预测启动记录

C3唯一best完整结果已检查（candidate3_result_report.md），质量未改善，停止继续扫描同类几何Loss系数。C4已独立登记candidate4_plan.json：冻结C1全部L/F物理草稿，仅学习分离轴与每轴节点rank。新增可训练参数80517；C1原241547个F参数均冻结，不清零原读出层。关系头复用C1第一头的隐藏/消息权重，新的rank和pair-axis读出清零，fresh AdamW/seed0。

NN输出一次关系图，供复制的solver_relations.py使用；与repair85/solver.py相比仅新增可选固定输入关系，原3次形状求解+1次投影、.18秒、原公开目标及无增强/无救援设置相同。两个节点rank决定每轴顺序，rank+ID并列规则保证每轴DAG；对称pair logits选轴。两预放置模块保留公开几何原生关系，其他关系由NN预测。DAG不保证与锚定位置兼容，必须保留最终硬检查与合法行回退。

固定Loss：每例axis CE均值+axis CE最坏对+signed rank order hinge最坏对+.25局部rank中心模仿，所有项/归一化见计划，反传/固定train/dev500/best/LR/早停同一函数。teacher只供监督，不在NN或最终求解输入中使用。初始原生axis prior为单调score/(1+abs(score))，rank为C1解码中心。

关系接口原五train机制短检完成，teacher关系5合法0回退；两较大例公开质量代理改善，但没有dev或泛化推断。C4预检第一次未通过非有限拒绝：被污染的是公开锁定teacher坐标，被decode覆盖，原输出有限门禁未检测到。已保留runs/20261102_candidate4_preflight/failed_source.zip与failure.json，增加显式teacher输入有限门禁，并登记公共锁定pair保护；未训练正式候选。

最终预检v2在同五train通过：真实Dropout/Adam eager/graph参数与动量差0；新图重建恢复下一步模型/Adam/RNG逐位一致；捕获无参数/Adam更新，非有限teacher（包括被公开decode覆盖的字段）在Adam前拒绝，L和C1 F保持逐位不变，初始几何等于冻结C1、pair logits对称。用时32.77秒，容差未放宽。

后台启动命令/PID见candidate4_launch.json，日志candidate4.stdout.log/stderr.log，正式训练由训练器登记4/8。上限1小时/100轮，原UTC2026-11-02T02:58:12全局截止不变，batch32 FP32、LR .001/.00025/.0001/.00005、原固定停滞/早停；best/latest保留完整Adam/RNG/order/cursor/seen/调度/累计预算。untrained.pt保留原始关系头，供必要无学习对照。

训练后candidate4_pipeline只评价唯一own-Loss best一次：完整逐例dev500（含关系传输/边构建/全部几何/硬检查），同100四阶段几何图；然后一次相同预算无学习固定原生关系dev500控制，若best即初始模型则只读复用相同评价。物理草稿严格相同C1，rawdev2000只读复用C1已有证据，不重复相同推理，不把复用写成新改善。未按Q选择checkpoint，不自动启动C5。

旧C1/C2/C3绑定源码哈希重新核验未变。活动C4关系模型/训练器/计划/流水线/评价器及依赖冻结，后续数值/目标变更另登记；测试封存，原split和冻结索引不变。
