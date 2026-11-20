# E153：MIB、增强题目编码与 S0 拓扑调整

2026-11-18 用户授权一次性准备一个完整方案，不做对照、不迁移历史权重。先后台处理数据，再修改模型。当前只授权准备，正式训练开关关闭。

数据复用 reference_workset_v6 的原训练前32000例、原train500/dev500，不更改验证划分。MIB平均面积派生任务保留公共固定规格；新缓存为全模块预测、MIB标识保留、17维基础属性＋16维结构角色＋完整端口token。数据处理身份、迁移记录和进度见 data_preparation_state.json，后台日志见 background/data.stdout.log。

全新 L0 为116043参数；完整模型733575参数。L0 与后层题目编码器都使用增强角色/多跳/端口/MIB输入，每个功能子层都获得增强条件。S0含门控分组移动和组内成员重排；三个双层单元各调用两次，跨单元只传布局。模型依然独立预测所有F/P模块的位置和形状。

训练流程准备为：从头训练L0 → 冻结本次新训L0的own-Loss-best → 从头训练增强题目编码器与S0/S1/S2 → 冻结backend own-Loss-best → 原dev500完整MIB后处理/评分。无历史checkpoint或Adam迁移。

后段L4/L8/L12监督权重.1/.1/.8；L4去直接overlap、reference为原系数的0.75；L8/L12使用公开任务和MIB、不加reference，L12加0.2的L8改动项。分组瓶颈和MIB一致性各以2B系数独立计入。L0保留reference、去直接overlap，加入公开分组/MIB目标。见 DESIGN.md 和 plan.json。

训练预算暂登记沿用scratch引擎：每阶段4小时/200epoch上限，原四档LR和平台早停。该设置是准备默认值，未把E152的十轮微调设置用于新scratch任务。尚未启动正式训练。

主验收：最终500/500硬合法、至少475/500题grouping全满足。MIB、贴边、Q和完整耗时独立报告。新任务与历史原面积/no-MIB分数不直接横比；test保持封存。

准备结果见 preflight.json；启动入口为 run_queue.py，但 authorization.json 的training_enabled必须由后续训练授权激活，准备过程不会自动启动。
