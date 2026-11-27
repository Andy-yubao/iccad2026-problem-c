# 候选2启动记录

从候选1唯一own-Loss best暖启动，不清零读出层；冻结L，same small F241547，same teacher16500，same slim3。新Loss与新AdamW/seed0已在candidate2_plan.json预登记，变更目标不是候选1普通恢复。正式槽位2/8，原全局截止UTC2026-11-02T02:58:12不变。

固定Loss为逐例最坏可靠teacher近邻对的原生分离轴/方向hinge + .25形状 + .25局部相对中心，取代绝对坐标模仿。teacher轴优势>=.05，目标优势上限.1、归一化delta截断8，均为一次登记，不扫dev。它是依据C1回退关联提出的可证伪机制，不能预称Q改善。

首次预检失败源于CUDA捕获不兼容，未正式训练，保留runs/20261102_candidate2_preflight/failed_source.py与failure.json。二轴gather改为等价where，原五train逐项Loss与state梯度逐位一致。预检v2：真实Dropout/Adam eager/graph参数与动量差0；捕获无参数/Adam更新；重新建图恢复后下一步模型/Adam/RNG逐位一致；非有限拒绝及L冻结通过。未放宽容差。预检34.24秒。

后台隐藏启动命令及PID见candidate2_launch.json，日志candidate2.stdout.log/stderr.log。初始化、固定dev/train500监测及实际训练均含在墙钟预算。candidate2_pipeline只接续本候选：自身固定Loss best唯一一次dev500逐例完整计时/硬检查、rawdev2000与固定同100四阶段图；不启动候选3，不改后处理器，不按Q换checkpoint。

C1绑定源码哈希已重新核验未变。teacher只读复用，测试未读。活动候选2源码/计划/模型/数据依赖冻结，后续修改须独立登记。
