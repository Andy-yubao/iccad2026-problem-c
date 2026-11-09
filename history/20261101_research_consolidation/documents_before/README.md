# ICCAD 2026 Problem C

**2026-11-01：[E95已按用户要求终止训练](experiments/E95_single_pass_corrector/runs/20261101_single_call_seed0/stop_20261101.json)。** 用户认为Loss异常，已安全停在5轮／2586步、累计23.8分钟，best1；优化器、RNG、调度和游标保留，训练及控制进程已退出、GPU释放。未进入后续评价或作图，STOP保留，不自动续跑；异常原因待诊断，测试封存。[状态](experiments/E95_single_pass_corrector/state.json)。

**2026-11-01：[E94已完成训练和500例验证](experiments/E94_posttrain_lr_reset/runs/20261101_lr1e3_seed0/qualification/report.md)。** P best36后训练从1e-3起，无保护，33轮/65.9分钟正常早停、best30；固定Loss0.323142优于E93 C，但加权Q1.025529仍未胜E90的1.022348，最大Q2.281252退化。500/500合法、4进程×2线程及同100图完成；高LR前三轮过冲，Q差区间跨零，不认定低初始LR是主因，不升级E90或追加训练，测试封存。

**2026-11-01：[E93两支验证完成](experiments/E93_matched_posttrain/runs/evaluation/report.md)。** C best10、G best16同例加权Q为1.026230／1.024601，均500/500合法，未证实优于E90的1.022348。G将完整dev2000草稿交面积5.950%降至4.908%，但外框、task项及最差Q存在取舍；各支完整dev2000、一次flash500、同100图及4进程×2线程核查完成。不升级E90，不自动续训，测试封存。

**2026-11-01：[E92完成资格评价，未过E90门槛，按约定终止](experiments/E92_two_stage_placer/runs/completion/report.md)。** 两阶段训练满2h、55轮／55262步，F own-best15；同dev500 Loss0.328326略优于E90的0.329100，但加权Q1.031269劣于1.022348，最差Q及覆盖尾部也退化。一次flash500均合法，完整dev2000与[同100草稿图](experiments/E92_two_stage_placer/runs/20261031_two_stage32k_seed0/qualification/draft_comparison100/index.html)已完成。未启动修正模型，每小时自动唤醒已删除、STOP已保存；权重和历史费用保留，测试封存。

[项目研究全貌报告（2026-10-30）](output/2026-10-30_项目研究全貌报告/README.md)：整体路线、图神经网络与注意力入门、关键实验结论及当前状态。

**2026-10-31：[E91已异常停止，失败诊断完成](experiments/E91_e74_32k_scratch/runs/failure_20261031/README.md)。** 41轮／41372更新，best41 dev Loss0.353941，平均较best32改善6.47%；但训练成员1032在best41严重失稳。被拒batch梯度19670超过10000，Adam前已拦截；加速前后同RNG逐位一致。保留剩余约37.6分钟，不盲目重启、不升级模型，无Q，测试封存。

**2026-10-31：[E89两分支均已完成核查](experiments/E89_spatial_neural_refinement/runs/completion/inspection.md)。** spatial19轮/best18，pooled13轮/best13，均正常早停；flash500加权Q1.024304／1.025208，均未胜E90的1.022348，稀疏157也无改善。空间/汇总对照Q差区间跨零，不支持二维信息的可靠收益，不升级基线；无活跃训练，测试封存。

**2026-10-31：用户采纳E90固定Loss作为后续基线，[E88稀疏续训](experiments/E88_sparse_training/README.md)已完成：11轮／5500步、23.6分钟早停，best仍为epoch0（E90起点），完整开发验证已完成，Q按相同模型复用原结果。** 从E90 best12分叉、基础16k与新增2268按80/20混合；LR重置1e-3、无保护期，保留Adam/RNG及历史费用；本流水线已结束，测试封存，无后续自动训练。

**2026-10-31：[E90辅助权重×3对照](experiments/E90_auxiliary_weight_placer/README.md)已完成14轮／7000步及验证，best12、flash500加权Q1.022348，500/500合法。** [完成报告](experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/inspection.md)：严重重叠尾部改善，整体/稀疏组仍劣于E74，Q差区间跨零，不升级基线；测试封存，无自动后续任务。

**2026-10-31：[E87 best11的Q评价已完成](experiments/E87_simple_supervised_placer/README.md)：固定开发500例、单进程2线程，500/500硬合法，加权Q 1.024171；尚未确认比E74或E42有显著变化。训练保持暂停，测试封存，不自动进入后续实验。**

**2026-10-31：按用户“先停训吧”，[E87新Loss训练](experiments/E87_simple_supervised_placer/README.md)已安全暂停在第14轮／7000步，best11、累计约71.5分钟；停训时尚未启动后续评价（现已另获Q评价授权）。** [停止记录](experiments/E87_simple_supervised_placer/runs/20261030_simple_seed0/pause_20261031.json)核实进程退出、完整优化/RNG断点保存，2小时累计上限保留，停止标记阻止自动续跑。[E88扫描](experiments/E88_sparse_training/README.md)已完成48,000例并选出2,268个新增稀疏成员，当前E87仍用原16k。后续E87→E88→[E89](experiments/E89_spatial_neural_refinement/README.md)均等待用户恢复，测试封存，自回归排除，E86暂缓。

**2026-10-30 E85已完成并核查，无活跃训练。** [E74协同空间修正检查](experiments/E85_cooperative_space_refinement/runs/completion/inspection.md)：dev2000交面积2.120%→1.942%，弱端口稀疏组3.309%→3.025%；最终Q差区间跨零，主要重叠差在训练前固定算子对照中已存在，稀疏空间组织问题未解决。两组均正常早停、各500/500后处理合法、200张四阶段图核验；不升级E42/E50，测试封存。下文旧阶段记录保持原结论。

**当前状态：本轮最小策略归因已完成，预设训练候选数用满，自动跟进已关闭；没有活跃训练。** [综合报告](experiments/E84_loss_control/runs/completion/strategy_attribution.html)：已学F在当前部署中有明确Q收益；新增几何目标包有明确低重叠草稿收益，但最终Q增益尚不显著；更强来源同样未显示可靠最终Q增益。规模、数据量和标签修复的独立贡献仍未查清，不把阶段结束称作研究问题全部解决，E42/E50继续冻结。

[E84完成审计](experiments/E84_loss_control/runs/completion/inspection.md)：同E81架构/起点/两来源/数据，仅以原G训练选模，30轮早停/best29、训练69.91分钟。草稿交面积4.816%（完整目标E81为2.491%，全部2000例更好），Q1.017497对1.016242的差值区间跨零；G-only外框更小、部分尾部更好，保留取舍。100张[四阶段图](experiments/E84_loss_control/runs/20261029_same_initialization_G_only_seed0/draft_comparison100/index.html)核验，三图实看；452冻结张量/唯一best及导出/初始监测分项一致性通过。

[E83完成审计](experiments/E83_strategy_attribution/runs/inspection.md)：完整已学F81相对仅确定性操作ΔQ−.011219，族区间[-.019212,-.005149]；换更强P74虽降低草稿重叠，却有尾部反例且Q区间跨零。两新终末不重跑，0训练更新，[200张四阶段图](experiments/E83_strategy_attribution/runs/index.html)核验。模块化阶段E81/E82/E84三个训练候选已用满，累计训练34604.40秒（约9.61小时），不是18小时总预算耗尽；所有恢复状态和预算保留，测试封存。

此前暂停期间的[四阶段图册](experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit/index.html)、[核查报告](experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit/report.md)及[五个研究问题](experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit/five_questions.md)保持原结论与证据：源布局质量影响残余错误；reference与派生约束存在冲突；弱端口稀疏问题尚未解决。以下研究过程记录保留其历史语境。

2026-10-29一期六个候选已完成并保留账本；用户新授权“长任务继续”，第二阶段研究进行中，已恢复30分钟跟进。二期先限3个新候选/6小时、每组2小时，旧预算继续计入18小时总上限。已归档15个历史实验并核对5,746个文件，只清理72个可重建编译缓存，保护全部权重与张量缓存。

当前开发候选[E77](experiments/E77_compact_refinement/runs/completion/inspection.md)：完整dev2000草稿交面积6.302%→2.205%，1999/2000例改善；500例均满足E74公开预测外框限制，精确空白继续下降。flash500加权Q1.017483，相对E42的worker配对区间[-0.013842,-0.003141]。最差单例Q2.163867劣于E42的1.927279，不能声称全指标占优或正式未见成绩，不自动替换冻结E42/E50。[100例reference／E77／E42图](experiments/E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0/draft_comparison100/index.html)。

[E76已完成复核](experiments/E76_bounded_refinement/runs/completion/inspection.md)：30轮早停/best29，flash500加权Q1.017693，相对E42的worker配对区间[-0.012021,-0.004259]，完整dev2000交面积降至2.733%。但相对E74有499/500外框扩大、均值约增加1.31%；保留其收益证据，不按Q直接升级基线。

[E77](experiments/E77_compact_refinement/README.md)已完成40轮、own-Loss best38；固定E76与E74预测外框，只学新增局部修正器。原预算保留；恢复训练阶段实测35.94分钟，另保守计入首次未知中断77.16分钟。训练、完整dev、一次flash500和100图已完成；最后报告的整数序列化错误仅通过保存结果汇总修复，冻结源码与终末评价不变。[完成状态](experiments/E77_compact_refinement/state.json)，[一期总账](experiments/E72_topology_posttrain/study.json)。一期结束时的关闭记录保留，本次新授权另开下述二期；测试封存。

新授权的[E78](experiments/E78_connectivity_posttrain/README.md)从E77优秀权重继续训练既有最后修正器，源E76和E74预测框不变。保存结果否定了简单压低原始分组/线长代理的充分性；新组合提前协调相容等面积MIB组形状，并加入固定模块间通道过填压力。五例初始形状协调增加4例重叠，负面证据保留；固定Loss/CUDA/公开规格/框/真实Dropout Adam恢复预检已通过，进入正式训练。[二期账本](experiments/E78_connectivity_posttrain/study.json)。一期结束记录不追溯改写，E77终末不重跑，测试封存。

[E78完成复核](experiments/E78_connectivity_posttrain/runs/completion/inspection.md)：MIB准备突变499→5例，flash加权Q1.015401、最大1.751561，但草稿交面积2.723%较E77的2.205%退化，不能只按Q升级。下一组[E79](experiments/E79_center_refinement/README.md)冻结完整E78及其尺寸，只新增中心修正器，以更强交面积项和一侧源次序保护尝试追回几何质量。五例预检通过；二期第二候选，每组2小时，旧预算保持。

[E79已完成复核](experiments/E79_center_refinement/runs/completion/inspection.md)：27轮/best26，完整dev2000交面积2.358%、相对E78全部2000例改善，500例尺寸逐字保持且不出原框。flash加权Q1.016383数值略退、对E78差值区间跨零，最大1.738220。保留取舍，当前没有活跃训练；先分析219/21/99等保存反例再决定最后候选。位移极少触及上限，不能直接归因于幅度不足；不重跑本组终末。

用户最新提出布局模型与修正模型独立化：[E80已完成复核](experiments/E80_modular_transfer/runs/inspection.md)，完整冻结E79三个修正器，仅将布局网络换为小型E42，框也来自E42；没有训练。总参数153.7万（约减少74%），flash加权Q1.017329、dev交面积2.785%，500例在自身框内；但最差Q2.200946退化，速度未见收益。支持模块化研究，尚不升级基线。已有三条比较只读复用，预算与训练槽未重置。自动跟进当前配置为暂停，本次保留。

[E81模块化阶段已完成复核](experiments/E81_multisource_refinement/runs/completion/inspection.md)：小型E42＋暖启动加宽F，部署171.3万参数，27轮/best26、训练64.77分钟。完整dev2000交面积2.491%、2000/2000胜E42，flash加权Q1.016242、最大1.825773，也优于E42的1.927279；500例包含自身预测框。对E80增益的Q区间跨零，主修复失败/救援仍存在，保留强开发候选，不自动升级冻结基线。

[E82已完成复核](experiments/E82_global_pressure_refinement/runs/completion/inspection.md)：28轮/best12，保留E81全部497旧张量，只学2304个公共通道输入权重，同一Loss开发均值下降.86%、交面积2.491%→2.445%，但Q1.018035、最大2.056040回退。164仍28/28主修复失败，救援后分组违反6→9；不升级此配置，主要保留E81候选。[第三阶段账本](experiments/E81_multisource_refinement/study.json)累计两个候选约128.82分钟、此前22680.53秒保留，全局18小时不变，本阶段还剩1候选但当前暂停，不自动使用。源码本地提交2f392f7；用户最新暂停已覆盖此前跟进确认事项，无待恢复请求。

## 冻结基线

| 基线 | 实现 | 同dev500结果 |
|---|---|---|
| draft＋flash | [E42](experiments/E42_joint_regularization/README.md)，96×12 best51、原steady | 加权Q1.025860 |
| pro | [E50](experiments/E50_anytime_quality/README.md)，共用E42 draft、质量优先5秒 | 加权Q1.012043，12例较flash回退 |

质量/时间预算分别比较；这些是MIB派生任务的自适应开发中性时间Q，不是正式比赛或独立盲测成绩。E42共享权重SHA256 `12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6`，唯一原路径和v5缓存保持原位。E74不自动替换冻结E42/E50。

## 保留的近期研究

| 实验 | 状态与用途 |
|---|---|
| [E69](experiments/E69_case_loss_statistics/README.md) | 固定dev案例难度、弱端口/稀疏子群和位置错配诊断 |
| [E71](experiments/E71_density_reference_curriculum/runs/inspection.md) | 紧凑草稿取舍对照；Q未胜出 |
| [E72](experiments/E72_topology_posttrain/README.md) | 首轮扩大模型的数值失败证据，5575步安全保存；长期研究总账 |
| [E73](experiments/E73_wide_robust_posttrain/runs/inspection.md) | 稳定宽模型完成；泛化差距和Loss/Q错位诊断 |
| [E74](experiments/E74_collision_32k_posttrain/runs/completion/inspection.md) | 当前草稿与flash均值更好的候选，预算停止、非收敛 |
| [E75](experiments/E75_collision_box_finetune/README.md) | 3374步数值保护失败，主干其他路径漂移证据 |
| [E76](experiments/E76_bounded_refinement/runs/completion/inspection.md) | Q/重叠改善，但外框扩大；完成并保留 |
| [E77](experiments/E77_compact_refinement/runs/completion/inspection.md) | 已完成复核：紧凑、重叠和Q均值改善，仍有尾部反例 |
| [E78](experiments/E78_connectivity_posttrain/README.md) | 二期：同形组协调＋锚点通道后训练，五例预检通过 |
| [E79](experiments/E79_center_refinement/runs/completion/inspection.md) | 完成：尺寸保持、重叠追回部分，Q存在取舍 |

归档E61/E65/E68/E70等对照与恢复方法见[迁移说明](history/20261029_model_research_consolidation/README.md)。[朱老师交付材料](output/2026-10-24_朱老师交付报告/README.md)保持原位。

先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。本聊天长训练授权记录在实验方案，不改写通用协议。独立环境`.conda/envs/iccad2026`，数据由[paths.json](protocol/paths.json)解析；冻结成员保持原件，checkpoint/数据/环境不入Git，布局图统一visualization，不自动启动评分界面。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [官方题目](references/official/problem_statement.md)
