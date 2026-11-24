# E159 四头每头32维的三层初稿生成器

2026-11-21，用户要求准备 E158 的加宽对照。仅完成源码和配置准备，正式训练未启动。遵循当前 AGENTS：不运行防御性测试、启动试验、模型试跑或computer-use，不复制旧preflight及其passed记录。

| 项目 | E158 | E159 |
|---|---:|---:|
| attention块数 | 3 | 3 |
| attention头数 | 4 | 4 |
| 每头维数 | 16 | 32 |
| 总隐藏宽度 | 64 | 128 |
| 每块FFN | 64→192→64 | 128→384→128 |
| 最终读出头 | 64→64→3 | 128→128→3 |
| 最终布局输出次数 | 1 | 1 |
| 参数量 | 170415，E158已记录值 | 667631，按源码结构公式计算 |

FFN继续保持3倍隐藏宽度。公开17维模块特征、16维结构角色、多跳编码、完整引用端口、静态关系bias、MIB、局部图聚合和端口静态key/value规则保持不变。输入投影及读出头随隐藏宽度配套变宽，不增加中间布局输出或几何反馈。

总Loss完全沿用E158：reference1.0 + 0.25倍完整原重叠组 + 原wire/compact + 原grouping/boundary + F/P尺寸 + P位置 + 2B*MIB。原重叠系数A=0.08975162861857286，有效系数0.022437907154643215。grouping不改，不加入新夹挡、瓶颈或移动监督。全部模块包括F/P仍独立预测位置及形状。

训练、固定train500/dev500、best、LR、早停均使用相同总Loss。复用random_mib40k_v1 train36k、固定监测成员和CPU缓存，不重抽样，不重新预处理。scratch seed0、batch64/尾批32、FP32、原轴互换增强、AdamW及参数更新投影、四档LR/早停保持，累计最多4h/200轮。原正常训练中的数值门禁保持；本次不执行额外测试路径，也不增加试训入口。

只训练L0，完成后冻结own-Loss-best，再从已保存监测日志自动生成与E158 best58的比较。无额外推理、后段、后处理、Q评价或test。报告固定train500/dev500的总Loss、所有加权分项、reference中心/形状、原始重叠、各自best轮次/更新数/耗时，以及同题Loss变化。不按比较结果重选checkpoint。

准备源身份见 preparation.json，该文件只登记源码与配置，不代表已通过运行测试。参数量由40*w²+89*w+879在w=128处计算，没有实例化模型或测量显存/吞吐。provenance.json保留E158来源指针；历史E158权重、源码和结果不改。

正式入口：`python -u -B -m experiments.E159_role_initial128.run_queue --start`。当前authorization.json的training_enabled为false；准备不构成启动授权。正式启动时使用隐藏的python.exe进程并重定向日志，避免pythonw丢失标准输出。后台状态按queue_state.json、runs/e159_initial_seed0/heartbeat.json及progress.json记录。
