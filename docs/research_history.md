# 有价值的历史研究：结论、反例与剩余问题

2026-11-08：E114三组与E115已完成，用户选择4+12为后续主要基线（[v2](../protocol/research_baseline_v2.json)）。加权Q/尾部坏例、四模型训练用时及单例入口干预见[最新诊断](../output/supervision_q_analysis_20261108/report.md)；[4+12固定5例＋最差5例三相图](../visualization/E114_supervision/20261108_staged_three_phase10/index.html)。全500结果与旧实验证据保持，不重训、不读test。

整理：2026-11-05。按问题组织，而非把编号当进度。本文是现有证据的综合，不新增训练、推理、求解、评分或测试读取。每节区分观察、解释和限制；所有旧命令/预算只属历史。完整实验位置见[目录索引](experiment_catalog.md)。

## 1. 研究谱系与基本术语

2026-11-06边界变更：以下成绩保留各自历史MIB面积变换/罚分与数据身份；后续新实验改为原始面积、MIB忽略和独立随机36k/2k/2k集合（E108准备）。这属于任务/数据/评分口径改变，不能把前后分数差直接解释为模型收益，也不追溯重写历史结论。

早期E1–E25建立神经草稿＋几何修复、教师/原reference监督与数据版本；E30–E50研究工程、矩形交互、共享草稿和更强搜索；E51–E75追查Loss/几何/数据/规模/端口差异；E76–E85形成布局L＋残差F并做归因；E86–E100继续尝试空间修正、目标、关系和混合预算；E101整理交付与重训；E103–E107转向Loss对齐、扩容/稳定主干、代理与几何后训练。E102始终是独立图片兴趣路线。

这里的“草稿”是L或F的解码矩形，尚未H；“最终解”是所有后处理和独立硬检查后的输出。内部解析梯度、投影和MIB协调即使写在模型forward内，也必须单列，不能全部归为神经网络所学能力。

常见几何量：成对交面积会重复累计多重重叠；重复覆盖面积为总模块面积减占用并集；逐模块覆盖为该模块被其他模块覆盖的并集/自身面积；N10统计覆盖超过10%的模块；真实空白为外框减占用并集。不同面板/分母的百分比不可直接拼接。

## 2. 完整后处理不可被未经对照的预算假设替代

**最重要的纠正来自E97。** 首窗口给所有模型固定三步弱修复，少数低质量合法回退支配Q，导致对新网络的判断过于悲观。后来10臂各500例接回完整E85单线程后处理，全部硬合法、零行回退；保存的新旧神经草稿逐例差0。以原E85为例，弱配置Q1.503224，完整配置1.015922067；C3完整Q1.013332050，已优于同条件E85。候选排名也变了。

这支持“后处理强度/执行路径决定大量最终差异”，不支持“所有网络都没用”，也不能把跨线程/环境的细小时间差当单因子收益。完整修复仍约5次shape调用，网络未证明能把迭代量稳定压到3步。

E35更早的条件删除证据也重要：罕见救援关闭后，两例仍硬合法却退到Q≈10。不能按低触发率或平均耗时删掉救援。单模块删除会把预算释放给其他分支，差值不可相加；数值规范化/硬守卫也属于正确性边界。

证据：[E97补测纠正](../history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/fullrepair_singlethread_findings.md) · [E35结论](../history/20261026_flash_pro_consolidation/experiments/E35_optimizer_attribution/runs/20261022_attribution_report/conclusions.md)。

E50质量优先搜索也是重要基线：同E42草稿完整dev500，加权Q从1.02585963降到1.01204277，但完整均时4.511742秒，12例较常规退化，因此没有通过逐例不退化的严格联合要求。它展示更充分搜索的潜力，不能把收益算到网络，也不满足250ms目标。用户对“尊重草稿”的澄清是相对方向和意图邻接，不能只用小位移代替；目前没有校准完毕的正式意图指标。[E50完整500报告](../history/20261103_final_delivery_consolidation/experiments/E50_anytime_quality/runs/20261025_full500_review/analysis.md)。

## 3. 学习F有效，但草稿减碰与最终Q并不单调

E83旁路F的所有已学残差，保留原固定MIB/框操作：Q1.027461；完整E81为1.016242，差−.011219，预登记族调整区间[-.019212,-.005149]。这支持部署中的已学F有独立贡献，不只是固定操作；它不是移除后充分重训的消融。

E84同初始化/模型/来源，只将完整目标包G+4I+5E+.25A改为G-only：草稿交面积4.816%，完整E81为2.491%，2000例均完整目标更低；最终Q差区间却跨零，G-only实际框更小、部分尾部更好。不能将重叠作用直接升级为Q作用，也不能分别归因给I/E/A。

同一F换更强上游E74，平均残余重叠下降，但数百例变差；E83的P74+F81 Q1.018527未可靠胜E81。上游初始重叠和残余错误高度相关，却无已证实“超过阈值完全修不回来”；B/A≥1也不能证明框可装。

E82反例164尤其清楚：草稿交面积1.366%→1.337%，最终Q1.825773→2.056040，两者主候选均失败并走救援，分组/线长更坏。完整四阶段图比只看最终图更能解释变化。

证据：[E83](../history/20261101_research_consolidation/experiments/E83_strategy_attribution/runs/inspection.md) · [E84](../history/20261101_research_consolidation/experiments/E84_loss_control/runs/completion/inspection.md) · [上下游审计及反例](../history/20261103_final_delivery_consolidation/experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit/report.md)。

E85还说明“放进网络的操作”不等于学到的能力：cooperative加入8步解析几何下降，约95.9%的最终组间重叠差在训练前已存在；最终Q差−.000292的区间跨零。这个比例是描述性比较，且中途调度不完全一致，不能作严格因果分摊。[旧全貌报告第9节](../history/20261103_final_delivery_consolidation/output/2026-10-30_项目研究全貌报告/项目研究全貌报告.md)。

## 4. E97第二轮与E101：保留交付和重训身份

E97 R2C5从原C3暖启动小F（241547训练参数），保留E74，加入公开N逐例权重以及已登记group-aware/接触目标。dev500加权Q1.012685893、500/500合法、零行回退，完整均值204.464ms/P95 521.080ms，126例超过250ms。仍未达到Q<1.01或逐例250ms。

相对当轮C3 Q改善.000680390；超额Q中soft仍占约78.1%，平均shape次数5.222→5.158，没有显著少做几何工作的证据。R2C6联合L后4层与F，自身Loss-best停在epoch0，模型严格等C5并复用结果；这是有限预算下未超起点，不是联合训练普遍无效。R2C7用actualWH教师也未升级；C8仅预检未训练。

**E101有多份权重，不应统称“最终C5成绩”。**

| 权重/运行 | dev500加权Q | 已登记2000加权Q | 状态/限制 |
|---|---:|---:|---|
| 原C5，sha前缀028ea3aa | 1.012685893 | 1.018808539 | 后来补齐原权重2000；此面板已被重训曝光 |
| 首次64k暖启动，c1f0ce28 | 1.031581961 | 1.036082419 | L/F暖启动、教师来源变化，不是三阶段随机64k |
| 历史链路重跑，ca0465e4 | 1.018307314 | 1.025842192 | E42随机16k→E74后训练32k→F16k |
| 三阶段均64k | 无 | 无 | E42 step63561用户停止；后续未启动 |

三条完成的2000面板均2000/2000硬合法、零回退。重训没有复现原C5低点，不足以推断数据变多必然有害：来源、教师、阶段、路径/预算等同时改变。test2000属已曝光重复评价，不是独立未见确认，也不用于选权重。跨时期耗时仅描述，不用事后挑最快结果。

证据：[E97第二轮](../history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/round2_final_report.md) · [原C5补测](../experiments/E101_final_c5_solver/runs/20261104_original_C5_test2000/report.md) · [首次64k dev](../experiments/E101_final_c5_solver/runs/20261103_final64k_seed0/evaluation/dev_monitor500/report.md) · [首次64k 2000](../experiments/E101_final_c5_solver/runs/20261103_final64k_seed0/evaluation/local_test/report.md) · [历史链路重跑](../experiments/E101_final_c5_solver/runs/20261103_replay/report.md) · [停止证据](../experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/stop_verification_20261104.json)。

## 5. Loss与监督：阻力有证据，取消监督的普适结论没有

E62从E42 best51和真实Adam状态做五题单步反事实：完整G有4/5真实交面积增加，去center_local当步梯度后5/5减少，配平步长仍成立；Dropout/latest补充只3/5改善。说明该处中心项/全局clip有阻力，不能推广到长期/所有数据；shape在这组有帮助，不宜一并删除。[E62](../history/20261029_model_research_consolidation/experiments/E62_e42_loss_update_audit/README.md)。

E54无reference从后期权重以重置Adam、1e-3和长保护起步，train/dev一起恶化。五题诊断自由中心都被截到0，logratio越过clamp，输出梯度为0。这个失败至少包含参数域/截断死区和激进优化设置；不是无reference不可能的证明。[E54失稳](../history/20261026_flash_pro_consolidation/experiments/E54_reference_extremes/runs/20261026_no_reference_seed0/convergence_analysis.md)。

E70新任务Loss降低dev2000交面积至3.820%，但共同G变高；E71密度/reference课程使外框与真实空白改善，相对E70同dev500框缩3.195%，交面积又有取舍。已试过密度能量，不能把“再加密度”当未探索机制。[E70](../history/20261029_model_research_consolidation/experiments/E70_reference_task_loss/runs/review.md) · [E71](../history/20261101_research_consolidation/experiments/E71_density_reference_curriculum/runs/review.md)。

E87–E94形成后续固定十项Loss的历史：E90辅助系数×3改善部分覆盖尾部，flash500 Q1.022347517；E94从同起点重置高LR后固定Loss低了约2%，Q1.025528585仍未胜E90。没有证明“LR太低”是目标错位的充分解释；额外训练量/整套调度也不同。[E90](../history/20261103_final_delivery_consolidation/experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/inspection.md) · [E94](../history/20261101_research_consolidation/experiments/E94_posttrain_lr_reset/runs/20261101_lr1e3_seed0/qualification/report.md)。

## 6. 端口、稀疏覆盖与二维空间：已做过哪些具体尝试

E65 已完成每层 24 维模块对向量消息及屏蔽显式几何的 feature_message 对照，消息由两端隐藏状态与关系生成，再加权聚合回节点；不能将“关系进入消息内容”作为未尝试的新机制。固定 dev500 G：E64 0.368244、geometry 0.351731、feature 0.357486；几何相对 feature 的完整 dev2000 G 仅低 0.59%，共同早期更新数不占优。用户后续明确反馈 E65 与 E42 的 draft 观感差别不大、仍未达到要求，不能用相对 E64 的数值改善冒充视觉验收或整体升级。2026-11-08 的首稿持续模块对状态提案有跨块关系记忆、取消中间坐标等实际差异，但核心消息机制重叠，已撤回推荐，不推进该候选；首稿独立训练目标保留。[E65 源码](../history/20261029_model_research_consolidation/experiments/E65_geometry_messages/model.py) · [完成报告](../history/20261029_model_research_consolidation/experiments/E65_geometry_messages/runs/review.md) · [用户原话](../history/20261101_research_consolidation/experiments/E69_case_loss_statistics/README.md) · [本次复核决定](../output/first_draft_generator_proposal_20261108.md)。

E68补完整逐边端口、移除摘要、保留绝对权重强度，单seed组合未改善草稿：dev2000交面积8.426%，E64为7.146%、E65为6.241%。不支持该具体组合，不能否定所有端口表示；也不能认为补端口能直接解决无端口模块。[E68](../history/20261029_model_research_consolidation/experiments/E68_port_tokens/runs/review.md)。

E88扫描新增2268稀疏训练成员，从E90续训11轮早停，own-Loss best=epoch0，模型严格等父模型并复用Q。证明本次选模未获得新模型，不证明新增数据本身无价值。[E88完成状态](../history/20261101_research_consolidation/experiments/E88_sparse_training/README.md)。

E89冻结E90，训练16×16二维占用/卷积修正，对照参数匹配的pooled分支。spatial Q1.024304、pooled1.025208均未胜E90，二者差区间跨零；调度有历史迁移，有限网格可能漏细通道。不支持该具体二维信息组合有可靠Q收益，不排除其他空间表示。[E89检查](../history/20261101_research_consolidation/experiments/E89_spatial_neural_refinement/runs/completion/inspection.md)。

## 7. 关系预测与吸引排斥：局部正确不等于整体可用

E97固定关系接口本身劣于原生动态关系；学习关系头又劣于接口控制，应分开评价接口损失和预测误差。E100全体方向准确率高，但reference近邻/重叠对相比坐标启发式仅小幅变化；几何更新层内部减碰10.34%→8.09%，仍差于E98草稿，最终同完整后处理Q1.031556747 vs1.028555439。方向准确、被满足和最终布局质量是不同门槛。[E100完整检查](../history/20261103_final_delivery_consolidation/experiments/E100_relation_guided_placer/runs/20261103_relation_guided_seed0/evaluation_dev500/report.md)。

E99冻结C3的吸引排斥在调参面板有收益，R15完整dev500反而比无AR差.000553；另500个未参与调参的训练来源题配对差+.001751。两组草稿精确对齐，后者仍是布局模型见过的train，不是test。该配置不升级，方向已停止，不自动扫描。[dev验收](../history/20261103_final_delivery_consolidation/experiments/E99_attraction_repulsion/runs/research/20261103_R23_frozen_R15_complete_dev500/acceptance_review.md) · [训练侧复核](../history/20261103_final_delivery_consolidation/experiments/E99_attraction_repulsion/runs/research/20261103_R24_frozen_AR_train_probe500/report.md)。

## 8. E98/E103：离线统计、局部方向、实际续训必须逐层判断

E98原固定基底best58同完整后处理加权Q1.028555439；泛化增强best71用新固定计算深度后处理加权Q1.043560018。**这两数不能直接归因为增强退化，后处理口径已变。** E98旧调权与单步搜索已被用户放弃，不恢复、不采用其权重选优结论。

E103第一轮是同1000训练题×5成熟模型、600/200/200题目隔离。检查200排序准确率55.61%→57.98%，选稿混合Q仅降.001003；40题直接草稿下降在小步长改善.001043，大步长仍变坏，双步长门槛失败。用户后来明确授权直接训第一轮系数，原失败证据保留，不能伪称门槛通过。

| E98.depth_budget.v1，同dev500 | 加权Q | 平均Q | 混合Q | 完整均值/P95 ms |
|---|---:|---:|---:|---:|
| 增强best71，父模型 | 1.043560018 | 1.050714113 | 1.047137066 | 203.488 / 488.322 |
| 新Loss续训best29（32轮早停） | 1.040796990 | 1.052651602 | 1.046724296 | 203.902 / 484.682 |
| reference×.75 best8（11轮早停） | 1.039699828 | 1.053338179 | 1.046519004 | 203.047 / 482.878 |

三者最终500/500合法、草稿0/500合法。续训新Loss模块覆盖≤10%率42.49%→49.52%，但混合Q仅−.000413；reference×.75混合Q仅−.000618，平均Q均变差。没有等训练量原Loss续训控制，不把微小变化归为固定系数的因果收益，不按Q重选best。

用户后来将主问题纠正为**跨题目全局Loss–Q**。新2000不同训练题（排除旧1000），每题一稿、两模型各1000，1400拟合/600检查，样本等权非负回归。检查Pearson原Loss .2530、当前v1 .2576、全局拟合 .2585；Spearman .3250/.3392/.4563。跨题差异保留，未按题内排序；相关改善不代表对同题草稿的有用梯度。共线性下系数0不意味着物理项无用，拟合系数未投入新训练。旧pair refit只是搁置历史提案。

证据：[1000×5及方向](../experiments/E103_offline_loss_alignment/runs/20261104_bank1000x5_seed0/report.md) · [新Loss真实评价](../experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500/report.md) · [reference .75](../experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0/evaluation_depth_dev500/report.md) · [全局统计](../experiments/E103_offline_loss_alignment/runs/20261104_global_independent2000_seed0/report.md) · [旧pair提案](../experiments/E103_offline_loss_alignment/runs/20261104_pair_enhanced_current_v1/report.md)。

## 9. 扩容和深层稳定性：E95 → E104 → E105

E95的局部消息把未归一化隐藏状态聚合再残差加入，五题latest模块token RMS达120606.70，输出却仍有限；同权重只改用现有norm后全token RMS约60756→93.86。不同token统计不可混算，但局部前置归一化有直接机制支持。[E95诊断](../history/20261103_final_delivery_consolidation/experiments/E95_single_pass_corrector/runs/20261101_diagnosis/report.md)。

E96压缩端口的一次修正dev2000交面积5.4420%→1.0854%、平均N10 28.229→1.354，但flash500 Q1.026523，仍不是仅凭减碰可升级。[E96评价](../history/20261103_final_delivery_consolidation/experiments/E96_port_compression/runs/20261101_centroid_bbox05_seed0/evaluation/report.md)。

E104同时扩大16k→32k、96×12→128×16（3,463,747参数），step20350触发Loss门禁后，用户授权local-prenorm.v2迁移，保留参数/Adam/RNG/游标/预算，重建best/平台。82轮早停best81。它包含路径修复，不能称纯scaling实验。

E105从头固定完整pre-norm、四路残差门控、SwiGLU，128×16、3,490,819参数；原32k/增强/Loss/调度保持，61轮早停best58。相对E104是结构组合且训练历史不同，不能归因给单一门控或FFN。

| 固定深度同dev500 | dev原Loss | 加权Q | 平均Q | 混合Q | 完整均值/P95 ms |
|---|---:|---:|---:|---:|---:|
| E98增强16k/96×12 | .375844625 | 1.043560018 | 1.050714113 | 1.047137066 | 203.488 / 488.322 |
| E104修复32k/128×16 | .337808010 | 1.032266480 | 1.042627090 | 1.037446780 | 198.871 / 470.170 |
| E105固定新结构32k/128×16 | .331486430 | 1.028161120 | 1.038349240 | 1.033255180 | 202.782 / 488.381 |

表中E104/E105数值按各自报告精度；三者最终500/500合法、零回退，草稿0/500合法。数据/宽/深/路径并未完全拆开，单seed与自适应dev不支持scaling指数或稳定泛化结论。E105是近期有用的布局器来源，尚未在与原C5完全统一的部署协议下决出优胜关系。

证据：[E104迁移](../experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0/repair_20261105/README.md) · [E104终末](../experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0/evaluation_depth_dev500/report.md) · [E105终末](../experiments/E105_gated_prenorm_placer/runs/20261105_gated128x16_seed0/evaluation_depth_dev500/report.md)。

## 10. E106后处理代理：数值拟合有少量信号，局部方向未合格

原train内新5000题×5稿=25000，三成熟来源原稿＋一对局部正负扰动。按题3000/1000/1000；全部用固定深度后处理生成真实标签。128×8图网络2,105,420参数，输入公开问题与草稿，不读reference/分母/来源ID。Q值误差＋局部对称斜率误差＋.1辅助误差，尺度仅fit。

26轮/13000更新正常早停，唯一best7；check Q RMSE .090106 vs fit常数基准 .094803，Pearson .2937。**局部方向准确47.06%，斜率Pearson .014，方向MSE还略差于零斜率。** 预测斜率RMS仅真实的.559%，不能把弱跨题相关当有用几何梯度。

总验证Loss下降27.47%，约98%改善来自辅助项；fit中5题占零斜率方向损失95.55%。部分小扰动Q大跳变可能涉及后处理分支，但未查路径不能断言标签错误。不能事后删高尾、按check调权或换checkpoint。尚未做真实梯度下降后的新后处理检查，更未训练布局器。

价值是保留完整标签与失败模式，说明未来目标必须同时衡量数值、局部响应和真实可实现更新；这版代理暂不接回布局训练。方向是否能通过监督/尺度/平滑改进仍未解决，不因此全盘否定可微代理。

证据：[完成检查和图](../experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/result_review/report.md) · [实际训练结果](../experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/critic/results.json)。

## 11. E107几何后训练：严格保护停止，不是已证实全局发散

从E105 best58、reference降到10%，主目标精确重复覆盖/模块尾部/真实空白，配合软保护/线长/漂移。接受22次更新，第23批在Adam前拒绝：单例49.70237 vs父eval3.766752，门槛37.667519；批均5.30584与grad482.23均未触发绝对门，所有值有限。STOP、best/latest和现场保护，未完成一轮、未终末评价。

保存22步模型的失败题eval总Loss3.55588，低于父3.76675，重叠下降而空白上升；不支持持续确定性崩坏。原失败未保存单例全部分项/拒绝前mask，无法精确重放49.70。Dropout学生与eval父比较、比例平方软保护的陡峭尺度是优先怀疑，不是已完成因果证明；只降LR不能消除这些差异。

随后“冻结小残差、父稿总偏移2%/.02上限”建议**已被用户否定**，不能当现行路线。最新提案转向公开能量、真实几何梯度、正定预条件器与有限展开；无reference、无固定父稿总位移限制。它仍未实施，局部下降公式不提供全局最优、可行性或训练成功保证。

证据：[停止报告](../experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/abort_report.md) · [只读诊断，末尾建议已被后续纠正](../experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/diagnosis_and_revision_proposal.md) · [最新公开能量提案](../output/reference_free_optimization_proposal_20261105.md)。

## 12. 工程与研究纪律也有可复用的反例

E30的几何微调在2/3种子均值改善，额外开发确认均值也改善，但CVaR/最大Q和极端时间退化，因此没升级。它提醒均值/P95/最坏5%是不同维度，不应继续挑种子。[E30确认](../history/20261026_flash_pro_consolidation/experiments/E30_stable_fast_geometry/runs/20261022_research_summary/analysis.md)。

E31同题微基准缓存初始化2.558→.722ms，实际主流程0/33命中，不支持端到端收益；E101双CUDA图实际更新短检约205→102ms，浮点次序变化已登记，不能拿五题倍数外推整轮。E104/E105完整耗时仍主要由后处理承担。[工程协议](gpu_training.md) · [E31](../history/20261026_flash_pro_consolidation/experiments/E31_conic_workspace_reuse/README.md)。

## 13. 尚值得判断的问题（提案，不是自动执行队列）

| 问题 | 现有最强支持/反例 | 仍缺的辨别证据 |
|---|---|---|
| 能否在保住质量的同时减少后处理？ | 完整H影响巨大；已有网络没显著少做shape | 固定质量基线下逐项简化的可比证据，不能先假设三步足够 |
| 能否用公开能量代替reference组织信号？ | E54含截断死区，未证明不可能；密度已试过 | 公开初始化/梯度通路、无学习优化控制、相同步数学习增益 |
| 可微代理能否提供真实方向？ | E106跨题弱预测，但局部近零 | 高跳变路径/目标支配诊断，预登记方向验证与真实更新 |
| 结构/数据扩大是否可重复改善？ | E104/E105固定深度链路改善 | 路径/历史/多因素控制与独立重复；当前无scaling结论 |
| 软约束/拓扑比重叠更关键吗？ | C5超额Q以soft为主；低重叠仍可能救援失败 | 利用已存trace区分关系接口、形状可行性和分组断裂 |

这些问题不意味着应该全做。下一模型应根据用户目标选择最有信息量的一步，明确支持与反例、代价和停止条件；不重启废弃搜索、不读取封存测试、不把旧开放授权当新无限研究预算。
