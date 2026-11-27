# E120 Node12-Collision on E119

2026-11-09用户已确认精确公式并授权准备后后台启动。保留E119两层初始全局图交互、共享64→64→3头、12层几何主干和状态反馈/连接与分组聚合；仅将18维逐对MLP替换为每层动态Node12，经独立12→64投影注入Q/K，V保持。tau=.25、每头缩放初值.1可学习、精确分离压力/log1p/连续几何反传，完整公式见formula_spec.md。约725619参数，实施核对后登记；Attention和准确碰撞计算仍O(N²)。

单组node12/scratch seed0/36k/.2L4+.8L12，原始面积/无MIB/全模块预测，固定规格由后处理满足。复用完整共享CPU静态缓存；动态碰撞逐层计算，不重复准备36k。保留原FP32/batch32/增强/AdamW/LR/早停与门禁，每组总计4小时或200轮。原E119 best55与其评价只读复用，不重训、不按Q重选、不读test、不新增图。

同原五个训练成员必要实施检查，含一次可丢弃受保护更新；正式训练重新初始化。从独立隐藏Windows任务运行，入口background.py，控制器run_queue.py。顺序train→冻结唯一Loss-best→原固定dev500/E98.depth_budget.v1；所有坏例/硬合法/Q和完整均值/P95/最大/超250ms保留。失败或STOP/QUEUE_STOP不推进、不自动重试；其他旧STOP保持。

状态queue_state.json；日志launch_logs/train.stdout.log和train.stderr.log；验证日志evaluation.stdout.log和evaluation.stderr.log；运行runs/node12_seed0，评价evaluation_no_mib_dev500。
