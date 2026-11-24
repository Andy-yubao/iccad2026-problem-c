# E160 静态关系消息初稿生成器

2026-11-22：用户授权准备后直接后台训练。以 E159 为对照，只在三个 attention 块的 value 更新中新增接收者相关静态关系消息；不增加按跳数分开的聚合，不生成中间布局。

每个接收模块根据自身、发送模块或端口的隐藏状态，以及原有公开静态关系，决定收到的消息内容。关系输入保留原 pair7（直接连接权重、连接存在、同组、最短路、可达、自身、MIB），再加入发送者模块／端口类型2维。原角色16维、attention bias、多跳编码、邻接／grouping／MIB局部聚合均保持。具体公式见 [DESIGN.md](DESIGN.md)。

| 设置 | E159 | E160 |
|---|---:|---:|
| attention→FFN 块 | 3 | 3 |
| attention头数／每头维数 | 4／32 | 4／32 |
| 隐藏宽度／FFN宽度 | 128／384 | 128／384 |
| 关系消息秩 | 无 | 8 |
| 参数量 | 667631 | 677087（源码公式计算） |
| 布局输出次数 | 1 | 1 |

数据和监督完全复用 E159：random_mib40k_v1 train36k、固定 train500/dev500及共享CPU缓存，保留MIB输入和修正标签；reference系数1.0、完整原重叠组×0.25，其余原Loss与系数不改。grouping不重新设计。训练、监测、best、LR和早停统一使用相同总Loss。

scratch seed0，batch64／尾批32，FP32，原轴互换增强、AdamW、梯度及参数更新控制、四档LR和早停；累计4小时／200轮上限保持。新参数初始化隔离随机数消耗以保留共同分支的初始权重序列；关系输出投影初始缩放0.01。单seed且增加9456个参数，本轮结论只对应这一具体静态关系消息实现。

只训练 L0，正常完成后冻结唯一 own-Loss-best，并从保存的监测记录自动生成 [与E159的比较](comparison_e159/report.md)。没有后段训练、后处理、Q或test。比较包含总Loss、分项、同题差异、完成轮数与耗时，不额外推理或按比较重选权重。

本次按 AGENTS 不执行防御性测试、试训或启动验证，不实例化模型测参数、不做显存／吞吐探测。preparation.json只是源码与配置清单，不是测试通过记录；训练自身沿用的图捕获、固定监测和数值门禁保持。

正式入口：`python -u -B -m experiments.E160_static_relation_message.run_queue --start`。后台输出在 `background/queue.stdout.log`、错误在 `background/queue.stderr.log`；训练状态和逐轮监测Loss在 `runs/e160_initial_seed0/heartbeat.json`、`progress.json`，更新日志在 `steps.jsonl`。实际启动记录见 `background/launch.json`。
