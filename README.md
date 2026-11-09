# ICCAD 2026 Problem C

2026-11-08当前展示：[E114 4+12三倍预算后处理最差10例](visualization/E114_supervision/20261108_budget3x_worst10/index.html)，左reference／中间滑块1–12层真实draft／右固定本次后处理，编号与G分组色、F/P及边界状态统一、无MIB。新评价已500/500硬合法，混合Q1.021638，完整均时341.8ms/P95 1027.3ms，82例超过500ms；原结果和权重保留。[全量对照报告](experiments/E114_supervision_placement/evaluation_budget500ms/output/comparison_report.md)。

2026-11-08 E116已完成：[首稿结构对照复核](experiments/E116_first_draft_comparison/result_review.md)。原四层/新版均正常早停，best dev500 Loss为0.468609/0.470894；新版高0.488%，记录训练更新更快，但重叠惩罚较高。仅既有训练日志，无新增Q/后处理/test，不自动续训；E114主基线及E101交付保持。

2026-11-08最新：用户选择E114 staged（4+12、best48）作为后续主要研究基线，替代E109 A的主要地位；旧身份保留，E101交付不变，未来默认16k不改，不启动新训练。见 `protocol/research_baseline_v2.json`。当前展示改为左reference／中间滑块切换1–12层真实draft／右固定后处理结果，固定5个训练例＋已完成dev评价最终Q最差5例，同色同尺度，不自动读test或重评整套；见 `protocol/three_phase_display_v2.json`。E114/E115已实际使用轴交换P=.5、Dropout .1/.05及AdamW decay .0003，不能记录为“未做增强”。


2026-11-07最新：用户确认本次两组均用36k，已授权E114已有4+12恢复→[E115九层1+9大首头](experiments/E115_first_head_supervision/README.md)串行后台训练。原4+12累计预算保留，前两组已完成模型不重训；未来默认16k规则不变。[本轮队列](experiments/E115_first_head_supervision/queue_state.json)。


2026-11-07最新：E114训练已按用户要求暂停。末层/密集监督已正常早停（best59/best52），第4/12层组在数据准备阶段step0保存退出；只验证前两组冻结best，测试封存。[暂停核验](experiments/E114_supervision_placement/pause_verification_20261107.json) · [两模型同五例逐层查看](visualization/E114_supervision/20261107_two_models/index.html)。以下后台启动说明为历史阶段。

2026-11-07数据更新：E114第4/12层监督之后的新实验默认训练 **16000**，从当前36000训练池中固定随机无放回抽取、冻结后复用。当前E114三组继续36k。[规则](protocol/research_training16k_v1.json) · [数据处理入口与说明](docs/datasets.md#后续16k训练子集)。开发集及封存测试不变，本次仅准备数据入口，不启动新实验。

2026-11-07最新：用户已授权后台启动[E114三组小模型监督位置训练](experiments/E114_supervision_placement/README.md)，terminal→dense→staged串行；实时状态见该实验queue_state.json。以下“仅准备”记录为此前阶段；E112/E113暂停不变。


2026-11-07最新：用户要求仅准备 [E114小模型监督位置对照](experiments/E114_supervision_placement/README.md)，不启动训练。三组64维/12图块、612307参数，末层/密集/4与12层监督；采用新random40k、无MIB和全模块预测。五例计算检查通过、零参数更新，训练开关关闭、STOP保持。E112/E113仍暂停。


公开问题 → 布局网络 L → 可选神经修正 F → 几何后处理 → 独立硬检查与质量评价。项目已完成多轮布局、修正、Loss、结构和后处理研究；竞赛交付方法保留 **E97 第二轮 C5＋原完整后处理（E101）**。后续独立研究不自动替换交付。

## 接手阅读顺序

1. [接手导读](docs/START_HERE.md)：任务、核心判断、最值得读的证据与常见误判。
2. [执行协议](docs/protocol.md)：当前数据、信息、训练、评价和授权边界。
3. [工作区结构](docs/workspace_structure.md)：源码、权重、有效缓存与历史路径。
4. [有价值的历史研究](docs/research_history.md)：按研究问题组织的结果、反例和未决问题。
5. [当前状态](docs/current_status.md) · [全部实验目录索引](docs/experiment_catalog.md)：按需进入原始报告。

## 后续实验新约定（2026-11-06）

2026-11-07：用户已选 **E109方案A：共享预测头＋逐层监督（best epoch61）** 为后续架构研究的重要基线。[冻结身份](protocol/research_baseline_v1.json) · [同五例逐层对比](visualization/layer_comparisons/20261107_same_train5/index.html)。其中关键子网络加深已获本次明确授权并登记为E112/E113：8图块、总参数接近A，两组串行训练后自动验证和五例展示。[后台队列](experiments/E112_deep_geometry_budget/queue_state.json) · [自动更新展示](visualization/critical_budget/20261107_pair/index.html)。E101竞赛交付不变，其余未选提案不自动实施。

后续新实验保留原始面积与全部题目规格，**不再施加MIB约束，也不对MIB软违规罚分**；不做平均面积或同形变换，其他约束照常保留。原始数据、官方规则、历史实验和E101交付口径不改；新研究分数单独标注。详见[新协议](protocol/research_no_mib_v1.json)。

新研究数据基底为独立随机 **40,000张（36,000训练／2,000验证／2,000测试）**，从原始全量重新划分、抽样，不沿用旧MIB筛选或旧split；后续默认训练改为上述固定16k视图。E108数据准备已完成；训练授权按各实验登记，测试继续封存。[数据准备记录](experiments/E108_random_reference40k/runs/20261106_random40k_v1/state.json)。

## 当前状态

核对截点：**2026-11-05 15:58（北京时间）**。当时未发现 python/pythonw/conda 进程；这是快照，后续执行前须再核查。[原始核查](history/20261105_workspace_handoff/process_snapshot.json)。

| 研究 | 已核实的状态 | 入口 |
|---|---|---|
| E101 交付 | 原 C5、首次64k暖启动、历史链路重跑均有完成报告；另一个三阶段64k链路已按用户要求停止 | [交付与各权重身份](experiments/E101_final_c5_solver/README.md) |
| E98 | 调权搜索已停止并放弃；固定基线和增强best71仍是证据/来源模型 | [停止边界与基线](experiments/E98_loss_weight_learning/README.md) |
| E102 | 2026-11-06 22:25用户停止于25614步，完整状态已保存，进程退出；全部STOP保留，不自动恢复 | [续训登记](experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_resume5_seed0_launch/README.md) · [后台状态](experiments/E102_reference_image_generation/runs/20261106_semantic_v21_bf16_b4_resume5_seed0_launch/state.json) |
| E103 | 离线拟合、全局统计、两个续训候选及固定dev500评价已完成 | [结果导航](experiments/E103_offline_loss_alignment/README.md) |
| E104 / E105 | 正常早停，唯一Loss-best及固定dev500评价完成 | [E104](experiments/E104_scaling32k/README.md) · [E105](experiments/E105_gated_prenorm_placer/README.md) |
| E106 | 25000草稿标签及单个代理训练完成；方向检查不支持直接用作布局Loss | [完成检查](experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/result_review/report.md) |
| E107 | 22次更新后第23批触发相对单例Loss门禁；STOP保留，未做终末后处理 | [停止与诊断](experiments/E107_geometry_posttrain/README.md) |

**2026-11-05整理本身不新增实验、不恢复旧运行、不开放测试；2026-11-06新增授权仅为上述E108数据准备。** 已完成授权不变成永久运行许可；历史文档中的“启动”“下一阶段”“自动唤醒”和命令只记录当时语境。

## 阅读结果时必须保留的区别

- 草稿Loss、草稿几何、最终硬合法、最终Q和完整耗时是不同结果；更少重叠不保证更低Q。
- E101的历史完整后处理与E98固定计算深度 `E98.depth_budget.v1` 分开比较；不能拼成无条件排行榜。
- 大多数成绩来自多次暴露的开发面板和单seed。E101已登记test2000曾重复评价，不称全新未见测试；其他实验不因此获得测试权限。
- 历史主线使用公开MIB平均面积派生任务，历史E102使用原始面积；后续新实验使用原始面积且忽略MIB，数据与评分身份分开。自由reference和评分分母不作网络输入。
- Q<1.01及逐例250ms的原目标未被整体实现；均值小于250ms不代表尾部达标，也不代表正式比赛成绩。

独立环境：`.conda/envs/iccad2026`；数据位置由 [protocol/paths.json](protocol/paths.json) 解析。[数据版本](docs/datasets.md) · [GPU工程](docs/gpu_training.md) · [Git约定](docs/git.md) · [本次整理清单](history/20261105_workspace_handoff/README.md)。
