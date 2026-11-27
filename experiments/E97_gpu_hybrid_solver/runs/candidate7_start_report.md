# 候选7归一化几何网络启动记录

C6完整结果及五train只读激活/梯度诊断已完成，见candidate6_result_report.md。C6负饱和导致主体梯度约1e-7、主要只剩输出bias，不能当作容量无效的证据。C7为独立架构/数值路径，不修改C6源码或证据。

C7冻结原C1 L/F，仍4个width192/heads4 RectangleBlock、同E74块/input/state_embed初始化；新增各块后LayerNorm，读出为LayerNorm+fresh Linear192/SiLU/zero Linear3，不再复用旧E74隐藏读出。新增适配器参数1,974,355，初始函数严格C1。没有关系头、新端口token或CPU后处理增强；原动态原生slim3仍3形状步+1投影/.18秒。

Loss逐字相同C1/C6：teacher center1+shape.25+near-relative.25。同函数用于反传/固定train/dev500/best/LR/早停，未扫dev系数。训练时每轮增加原五train重复batch32的eval/no_grad读出激活记录，无优化器/参数/RNG变更，不驱动checkpoint、LR或早停。

预检37.58秒通过：真实Dropout/Adam普通/图参数与动量差0；新图重建恢复下一步模型/Adam/RNG逐位一致；非有限teacher在Adam前拒绝、L/C1 F逐位冻结、初始输出等价C1。第二次实际更新后读出pre-SiLU RMS=.670/.641/.663/.637、正激活比例53.3%/52.2%/57.7%/47.9%；隐藏读出/第一qkv梯度范数.00945/.00366，确认主体有梯度。diagnostic RNG前后逐位一致。

峰值allocated1256.41 MiB。新结构原五train公开输入到draft统一120模块GPU图batch1一次为16.02/12.08/12.23/14.00/11.43毫秒，均值13.15/P95 15.62毫秒；图捕获.664秒。这是单次五例短检，不是完整服务或dev500达标证据，不与旧重复分布强行比较加速比。

后台命令/PID见candidate7_launch.json，日志candidate7.stdout.log/stderr.log；训练器实际登记7/8，尚剩1正式候选。fresh AdamW seed0、batch32 FP32/noAMP/TF32、LR .001/.00025/.0001/.00005，1小时/100轮、原UTC2026-11-02T02:58:12全局截止不变。完整best/latest/untrained保存Adam/RNG/order/cursor/seen/调度/累计预算。

已安排自身固定Loss唯一best一次原slim3完整逐例dev500、新raw2000、同100四阶段图；若best仍初始零读出，则剪去恒等适配器复用已完成C1终末/raw，不重复求解。物理几何学到变化时必须新raw诊断，不能沿用C4/C5冻结物理几何的复用规则。最终仍以完整Q/回退/P95和草稿几何判定，不把正常激活或Loss下降当成成功。

模型类为E97.normalized_geometry.DeploymentModel，model_config字段与旧类相似不代表可互换；归一化模块和head.3是权重身份的一部分。后续若暖启动需该类strict load既有模型，不能再次清零读出或重新初始化。C1-C7绑定源码已全部核验未变；测试封存，teacher/oracle不重生成/重求解，split/索引/唯一权重保护。活动模型/训练器/计划/评价器及依赖冻结，变更独立登记。
