# E115：九图块、大首层预测头、1+9监督

用户明确批准设计，并更正本轮两组均使用与E114已完成两组相同的36k数据。队列仅E114已有4+12恢复→E115新1+9；不使用16k子集，不改未来默认16k规则。准备检查通过后后台启动；错误或用户停止不推进，无自动Q/测试评价。

64维、9个独立图块，共享小头64→64→3用于初始第0次及第2～9图块读出；第1图块后用独立大头64→256→256→256→3，中间两层构成SiLU残差支路。大头148995参数，模型609682，4+12为612307。仅第1和第9层分别.2/.8完整Loss，全部层联合训练；真实初始第0次不监督。原始面积/无MIB/全模块预测、公开固定/预放置规格条件、后处理满足硬约束的边界保留。

先按E114原12块构造初始化，再裁去后三块，保证共同编码器、前九块、状态嵌入和共享小头逐位同seed0起点；新首头在forked CPU RNG内初始化，不推进全局Dropout流。大头继承原小头前64隐藏通道，新增输出列初始0；残差末线性层使用小非零.01初始化，保留原相对步长保护。它只按模块计算一次，不扩展模块对几何MLP。不把大头深度等同于更多跨模块消息传递。

独立环境：`python -B -m experiments.E115_first_head_supervision.preflight`为五例实施检查；`python -u -B -m experiments.E115_first_head_supervision.run_pair_queue`为仅一次的两作业后台队列。单独训练入口：`python -u -B -m experiments.E115_first_head_supervision.train --arm first9`。不要重复派发已运行队列；失败/停止后需针对性处理。新队列STOP/QUEUE_STOP位于本目录，并转发当前训练的STOP。

两组同train36000、固定train/dev各500、batch32/FP32、原轴交换、AdamW/全局clip/每模块.5%步长门禁、原四档LR/14轮保护/耐心3。每组原4h或200轮上限；E114恢复保留已消耗约513.41秒，E115为独立新预算，不冒称等工作量。阶段内latest恢复保持模型/Adam/RNG/排列游标/seen/调度/保护计数/累计预算，best按固定own-Loss dev500选取，不看Q回选。

日志：launch_logs/staged_resume.stdout.log 与 first9.stdout.log，对应stderr记录错误；queue_state.json为本轮队列实时状态。E114原queue_state.json及QUEUE_STOP保留，仅直接恢复已明确授权的staged训练，不重启旧三组队列，不重训terminal/dense。E112/E113和其余旧STOP保留。
