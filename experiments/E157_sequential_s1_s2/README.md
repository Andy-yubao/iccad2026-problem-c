# E157：依次训练 S1 与 S2

2026-11-20用户明确要求：S1训完冻结own-best，再训练S2；准备通过后后台依次启动。冻结E156 best79全前端（含E155 L0和S0编码器/单元），新S1编码器与双层S1 scratch，S1共享调用两次输出L5–L8。阶段结束冻结完整S1模型，再新建S2编码器和双层S2 scratch，共享两次输出L9–L12。冻结前缀始终eval/no-grad、无输入布局扰动；跨调用只传布局，保留原轴互换训练增强。

S1：0.2*P6+0.8*P8。S2：0.2*P10+0.8*(P12+0.2*D8to12)。P为完整公开任务：原A绝对重叠、线长/紧凑、原grouping/贴边、F/P规格及2B*MIB，reference与grouping瓶颈监督均为0。S2移动容差0.05sqrt(area)、logratio0.1，形变内部权重0.25，Huber beta0.1；输入L8重叠负担>自身面积1%的模块，移动/形变惩罚权重减至0.25。S1无移动惩罚。

新36k、固定新train500/dev500，batch64尾批32、seed0，四档LR和原平台早停/数值门禁；每个新阶段独立最多4小时/200轮，无旧Adam迁移。按固定dev500同一登记Loss选唯一own-best，不按Q选权重。每轮记录冻结L4、S1 L8及S2终端的同口径公开Loss、重叠、grouping、MIB与拓扑诊断。

两阶段正常完成后对唯一S2 best运行新dev500完整MIB后处理和独立硬检查，报告grouping≥475/500目标、MIB、贴边、外框>reference1.2倍、Q和完整耗时。评分是平均面积派生任务/原reference分母/中性时间本地研究Q，不是原面积正式成绩。test保持封存。

总日志background/queue.stdout.log；队列queue_state.json；阶段runs/e157_s1_seed0及runs/e157_s2_seed0。STOP/QUEUE_STOP/PAUSE_REQUEST会安全保存并阻断后续。
