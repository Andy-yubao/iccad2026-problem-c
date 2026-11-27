# E117 Node26

2026-11-09用户确认：4+12监督、原random40k的36k训练，检查通过后直接后台启动；训练成功后自动验证固定dev500。

每层用26维动态节点几何（16自身、8连接/分组均值、2有效标志）经独立26→64投影添加到Q/K，每头共享Q/K的可学习缩放初值0.1。V、连接/分组隐藏消息、状态反馈、共享预测头和12层64维4头主干保留，删除逐对18→24→4 MLP。Attention仍是O(N²)。四边顺序为L/R/B/T；logratio轴交换取负。邻居标志按归一化矩阵行和>0定义，连接自环按原图保留。

scratch seed0；兼容的原E114初始张量保持一致，新几何投影独立固定种子；不读旧权重。Loss=.2L4+.8L12，反传/监测/best/LR/早停一致。原始面积、无MIB、所有模块预测；固定规格仅作条件，后处理执行硬约束。

batch32/FP32、轴交换P=.5、Dropout .1/.05、AdamW decay .0003、原LR/门禁及每组4小时或200轮上限保持。仅本组，无Node26-Separate或基线重训。最多原E114五个训练成员实施检查，含一次可丢弃受保护真实更新；正式训练重新seed0初始化。

成功训练后仅冻结own-Loss-best，自动一次原固定dev500/E98.depth_budget.v1后处理及原硬检查，报告草稿Loss/合法性、最终本地无MIB Q、均值/P95/最大完整耗时和全部坏例。未采用三倍预算，不按Q重选权重。测试封存；失败或用户停止不推进、不重试。历史E114是同数据/目标的单seed参照，参数与算力不同。

后台入口：`python -u -B -m experiments.E117_node26.run_queue`（需设置CUBLAS_WORKSPACE_CONFIG=:4096:8）。

状态：`queue_state.json`；训练日志：`launch_logs/node26.stdout.log`和`node26.stderr.log`；验证日志：`launch_logs/evaluation.stdout.log`和`evaluation.stderr.log`。

停止：新建本实验`STOP`或`QUEUE_STOP`，批次边界完整保存并阻止验证；不更改其他实验旧STOP。完整训练状态在`runs/node26_seed0`，验证输出在其`evaluation_no_mib_dev500`。同五训练例12层坐标保存在`visualization/E117_node26/20261109_train5/layers.npz`，不额外推理dev/test。

2026-11-09恢复：用户明确修复并重启。静态图特征和Loss端口张量改用共享分块CPU缓存，首次仅构建缺失成员，后续加载复用；同五训练例全部字段逐位核对。模型/动态Node26/Loss/FP32更新不变；从原step0完整latest恢复，保留约898.82秒累计预算。原现场和源码保存在recovery_20261109，恢复日志单独放launch_logs/recovery1。改用独立隐藏Windows任务，避免命令窗口退出连带结束；原失败原因无堆栈，当前已确认原进程处在禁止脱离且关闭即结束进程的Windows Job中。恢复仅此一次，后续错误不自动重试。
