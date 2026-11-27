# ICCAD 2026 Problem C

2026-11-13：[E133 Flow Matching质量评价已完成](output/flow_matching_quality_20261113/report.md)。训练best239，原自动评价0例记录错误已保留；本次用户请求下独立评价原dev500/Euler8，500/500最终硬合法、0回退，混合Q1.157321，低于E119历史质量（1.021568，Q越低越好）；草稿重复覆盖10.136%，完整均时206.4ms。权重/冻结源码与原8h记录未修改，无重训/test/新增图；详细可比性和五例步数诊断见报告。

2026-11-12新增：[E133 Flow Matching](experiments/E133_flow_matching/README.md)已完成研究设计与同五训练成员检查，并派发独立后台任务。单组scratch seed0、96×6条件速度场/920643参数、冻结16k、纯FM Loss；训练成功后唯一Loss-best自动Euler8固定dev500及原后处理评价。后台总预算8h，原绝对截止不因零更新准备修正清零；当前状态见`experiments/E133_flow_matching/queue_state.json`，正式日志`launch_logs/20261112_seed0_prepared`。[设计与启动登记](output/flow_matching_20261112/README.md)。不读test、不新增图，原实验与研究/交付身份保持。

2026-11-12最新：按用户要求已暂停E13164×4（full完成93轮/46929步，完整状态保留，pure未启动），新增[E13296×6两组](experiments/E132_relation_input_96x6/README.md)，准备检查通过后后台full→pure scratch seed0；其余训练设置保持，无后处理/Q/test。暂停核验及新启动登记在[output/P0_scaling96x6_20261112](output/P0_scaling96x6_20261112/registration.json)。下方E131启动为历史状态，不自动恢复。

2026-11-12 E131最新：用户已授权按准备配置后台full→pure串行训练，并要求派发后不持续监听；仅本实验准备STOP归档释放，训练授权开启，本次无后处理/Q/test。独立启动登记在[启动记录目录](output/P0_relation_input_launch_20261112/registration.json)，实际派发信息dispatch.json，队列状态`experiments/E131_relation_input/queue_state.json`；下方仅准备说明保留为历史。

2026-11-12：[P0关系输入编码对照](experiments/E131_relation_input/README.md)已准备，full/pure均64×4、唯一标准读出，完整机制组268387参数、纯Attention组235331参数。固定五训练例的编码/CPU梯度/FP32batch32 CUDA图通过，零参数更新；训练授权关闭、STOP保留，本次无后处理、无dev/test张量读取。[准备记录](output/P0_relation_input_preparation_20261112/README.md)。

2026-11-11整理：用户要求experiments仅保留E119与E120；E98、E101–E118、E121–E130共29个实验目录已同盘归档至[history/20261111_experiments_consolidation](history/20261111_experiments_consolidation/README.md)（逐文件清单与SHA256核对，映射见manifest）。研究基线E119、Node12对照E120身份与E101交付不变；恢复归档实验须先明确授权。

2026-11-11：[E129/E130三处监督结果](output/supervision_s0_20261111/result_review.md)。两组正常早停并完成原dev500后处理，均500/500硬合法；混合Q1.039560/1.058783，未改善各自原基线。本次授权已完成，不自动追加实验。

2026-11-11新增授权：[E129初始头监督](experiments/E129_global2_s0ref/README.md)与[E130节点结构](experiments/E130_node12_s0ref/README.md)。固定0.1×S0纯reference＋0.1×L4原Loss＋0.8×L12（reference×0.8、其余原值），原架构/12几何块/共享头/36k/scratch seed0。短检通过后独立隐藏后台各训练→唯一own-Loss-best→原dev500/E98.depth_budget.v1后处理；错误/停止不推进不自动重试。进度`experiments/E129_global2_s0ref/queue_state.json`，登记`output/supervision_s0_20261111`；不读test、不新增图。

2026-11-11：[E125–E128监督首轮结果](output/supervision_ref_20261110/result_review.md)。四组正常早停并全部完成原dev500后处理，均500/500硬合法；统一减弱reference及早强后弱分配均未改善各自原基线混合Q。本次授权已执行完，不自动追加其他方向/seed，研究基线与交付保持。

2026-11-10新增授权：[E125–E128监督首轮](experiments/E125_global2_ref04/README.md)。原E119/E120架构各做reference统一.4/.4与分层1/.25，保留.2L4+.8L12及原物理项，36k/scratch seed0；原目标S0结果复用。五训练例短检通过后独立隐藏后台串行训练→唯一own-Loss-best→原固定dev500/E98.depth_budget.v1后处理。主进度：[queue_state.json](experiments/E125_global2_ref04/queue_state.json)，各自launch_logs；失败/停止不推进、不重试、不读test、不追加后续研究方向。准备与启动登记见`output/supervision_ref_20261110`。

2026-11-10：[scaling16当前对比与故障诊断](output/scaling16_recovery_20261110/comparison_report.md)。E122/E123训练和原dev500均已完成；E124在34轮/38832步触发梯度门禁，用户要求继续后仅一次完整状态隐藏后台恢复，原LR/门禁/累计预算保持。成功后自动原dev500，再错误/停止不重试。[恢复进度](output/scaling16_recovery_20261110/supervisor_state.json) · [主队列](experiments/E122_global2_scaling16/queue_state.json) · E124新日志`launch_logs/recovery1_20261110/`。

2026-11-09新增授权：[E122/E123/E124 scaling16](experiments/E122_global2_scaling16/README.md)，分别扩展E119/E120/E121为16几何层，保留2层初始全局交互与原预测头；三组scratch seed0/36k/监督.1L2+.1L8+.8L16，准备通过后独立隐藏后台串行训练，各唯一Loss-best自动原dev500/原E98深度预算。状态与日志：[队列](experiments/E122_global2_scaling16/queue_state.json)、各自launch_logs。失败/停止不推进、不自动重试、不读test、不新增图，原研究/交付身份保持。

2026-11-09最新授权：用户最终确认E121_residual_update研究布局状态S残差，原E119头仅生成S0，12几何块各自独立Linear64→3输出delta，S_k=S_(k-1)+delta、eta固定1；保留E119初始2全局图块/18维模块对MLP/状态反馈，非H残差、非共享delta头、非Node12。714615参数，新增2340。原随机delta初始化实际触发有限Loss/梯度门禁、零Adam；仅新层权重和bias初始缩放.01（非零），eta/公式/旧主干/LR门禁不变，失败原件归preparation_diagnostics。相同五训练成员缓存短检、精确递推和一次受保护更新通过，254个E119初始张量/CPU RNG对齐。仅单组scratchseed0/36k/4+12/FP32batch32/原增强LR门禁/4h或200轮；准备后独立隐藏后台启动，成功后唯一Loss-best冻结并自动原固定dev500/E98.depth_budget.v1。完整复用静态CPU缓存，不重训基线、不读test、不新增图，失败/用户停止不推进不自动重试，旧STOP保持；状态E121/queue_state.json，日志launch_logs。E119研究基线/E101交付不变。

2026-11-09最新授权：用户确认Node12具体公式并要求准备后后台启动，登记E120_node12_collision，单组seed0/36k/.2L4+.8L12/FP32 batch32/原增强LR门禁/4h或200轮；保留E119初始2层全局图交互、共享头和12层主干，将逐对18维MLP替换为逐层Node12、独立12→64 Q/K投影，V不变。tau=.25、精确四向脱离压力、log1p/计数归一化、连续项正常反传、每头lambda初值.1可学习，无B/C或碰撞图；实际725619参数。原五训练成员数学/轴交换/梯度/CUDA与一次可丢弃更新检查通过，复用完整36k/dev500静态缓存；正式scratch初始化。独立隐藏后台任务运行，成功后唯一Loss-best冻结，自动原固定dev500/E98.depth_budget.v1（非三倍预算）无MIB本地Q/硬检查/完整耗时；失败/停止不推进、不自动重试、不读test、不新增图。状态E120/queue_state.json，日志launch_logs，公式formula_spec.md；E119研究基线/E101交付及其他旧STOP保持。此前“精确公式待确认/不实施”边界已由本次明确授权覆盖。

2026-11-09最新基线：用户明确选择E119 global2 best55为今后主要研究基线，替代E114主要地位；冻结身份见protocol/research_baseline_v3.json。E119共712275参数（2层初始全局图交互＋原12层几何块/共享头），36k/4+12，末层dev Loss0.333128、混合Q1.021568、原固定深度后处理500/500合法；57轮早停/best55，未观察到train平台。E114/E109旧身份与E101交付保持。下一方向仅确认“保留E119初始生成器与现有预测头、模块对特征改为模块特征”；用户要求具体实验要求必须先问，Node12-Collision/注入方式/压力公式和归一化/数据训练评价设置均待确认，当前不得据此实施新模型或启动训练。仍全模块预测、固定规格由后处理满足；不从附件中引入固定模块mask/覆盖。

2026-11-09最新授权：E118原E114同架构1+12监督（612307参数）→其固定dev500→E119初始生成器前增加2层无草稿几何的全局Attention＋归一化连接/分组聚合（712275参数、仍原4+12监督）→其固定dev500，串行后台各seed0/36k/4h或200轮；已完成E114 4+12 best48与原评价只读复用，不重训、不用Node26。两组同原五训练例缓存短检和一次可丢弃受保护更新通过；主干初始226张量与CPU RNG对齐，FP32/batch32/增强/LR/门禁/无MIB/全模块预测保持。完整复用现有36k/dev500静态缓存；各唯一Loss-best冻结后用原E98.depth_budget.v1（非三倍预算），不按Q回选、不读test、不新增图。独立隐藏Windows任务启动，无需等到正式更新；错误/用户停止不推进、不自动重试。控制器E118/run_queue.py，实时E118/queue_state.json，日志各自launch_logs。E114研究/E101交付不变，其他旧STOP不动。

2026-11-09 11:09（北京时间）E117恢复已核实进入正式训练，已超过600步，CUDA初始化及数值门禁正常。已完成共享CPU缓存train36000/dev_monitor500，后续复用；改为独立隐藏后台任务。原完整状态和累计预算保留，正常结束仍自动冻结唯一Loss-best及原固定dev500验证。新日志experiments/E117_node26/launch_logs/recovery1，恢复证据recovery_20261109/startup_verification.json。

2026-11-09：用户明确Node26采用36k和4+12监督，五训练例检查（含一次可丢弃受保护更新）通过后后台启动E117。仅本组seed0、12层/64维/4头、647155参数，替换逐对MLP为每层26维节点几何Q/K投影；FP32/batch32/增强/原Loss与门禁、4h/200轮保持。训练成功后自动冻结唯一own-Loss-best，原固定dev500/E98.depth_budget.v1无MIB本地Q及完整硬检查/耗时，再存同五训练例逐层坐标；失败/用户停止不推进、不重试、不读test。状态和日志见 `experiments/E117_node26/queue_state.json`、`launch_logs/`。E114主基线/E101交付及其他旧STOP不改。

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
