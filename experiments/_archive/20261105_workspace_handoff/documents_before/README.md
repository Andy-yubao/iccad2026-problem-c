# ICCAD 2026 Problem C：最终交付

2026-11-03结束竞赛方法研究，最终采用 **E97第二轮C5＋原完整E85单线程后处理**，独立交付实现为[E101](experiments/E101_final_c5_solver/README.md)，不再增加竞赛候选、吸引排斥前置或自动唤醒。旧实验、旧输出及辅助脚本已完整归档到[历史目录](history/20261103_final_delivery_consolidation/README.md)，权重、有效缓存和原始证据均保留。

2026-11-04用户明确恢复E98并进一步授权[多轮Loss调权后台研究](experiments/E98_loss_weight_learning/dynamic_weight_learning/README.md)：舍弃旧pilot对新研究的指导作用，从原固定Loss best58重新开始；5组设计方向后5组固定种子随机方向，共10轮。每轮三支共用从冻结1000训练池抽出的100例后处理反馈，混合Q=(加权Q+平均Q)/2负责接纳。验证集Loss负责阶段内best/LR/早停，测试封存；新运行独立保存，不改变E101交付。状态见[state.json](experiments/E98_loss_weight_learning/runs/20261104_dynamic_mixedq_seed0/state.json)，停止方法见研究说明。

E98最新状态：用户已要求立即停止。多轮运行在第3轮control第306次更新处安全保存latest，best保留；托管进程已退出，STOP保持，未启动该轮后续分支、更后轮次或终末评价，不自动恢复。[停止核验](experiments/E98_loss_weight_learning/runs/20261104_dynamic_mixedq_seed0/stop_verification_20261104.json)。

用户随后明确授权优先准备并后台启动[E98泛化增强基线](experiments/E98_loss_weight_learning/generalization_baseline/README.md)，训练期间继续准备固定计算深度评价器和冻结父模型的调权流程。新基线从头初始化，96×12、原train16k及固定Loss，FFN Dropout=.10、残差Dropout=.05、50%轴交换；验证Loss负责best/LR/早停，不计算Q、不读测试、不自动启动调权。旧多轮运行STOP保持。

泛化增强基线现已正常早停71轮，唯一Loss-best为第71轮，train500/dev500 Loss=0.371603540/0.375844625。用户随后明确授权固定dev500后处理评价：冻结该best，使用新固定计算深度评价器单次运行，分别报告加权Q、平均Q、混合Q、硬合法及完整耗时；[评价状态](experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/evaluation_depth_dev500/state.json)。不按Q改选模型，不启动调权，测试仍封存。旧基线Q采用时间预算后处理，不能直接与本次Q作网络归因比较。

该dev500评价已完成：[报告](experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/evaluation_depth_dev500/report.md)。加权Q=1.043560018、平均Q=1.050714113、混合Q=1.047137066；最终500/500硬合法、零回退，草稿0/500硬合法。完整逐例均时203.488ms/P95 488.322ms/最大1437.873ms，157例超过250ms；同100例图已生成。train/dev Loss差距收窄至1.14%，但验证Loss绝对值比旧基线高5.88%，尚不能认定最终质量提升。

用户随后仅授权[单步削弱reference候选](experiments/E98_loss_weight_learning/single_reference_step/README.md)，并明确后台运行：从上述best71开始，position=1/1.2、shape=.25/1.2，其他8项不变、不做总和归一化。父模型不训练；候选按验证Loss选best/LR/早停、最多15轮，保留原泛化手段。单次训练反馈100/1000后，对唯一冻结候选做一次dev500，与已完成基线对照后停止；不追加方向，不使用旧搜索权重/系数/选优结论。状态见[state.json](experiments/E98_loss_weight_learning/runs/20261104_weaken_reference120_seed0/state.json)，[训练日志](experiments/E98_loss_weight_learning/single_reference_step/stdout.log)。


用户最新已要求停止并放弃E98调权方向；train500运行在第3轮plus第1149步安全保存，进程退出，后续分支未启动，STOP保留，不自动恢复。[停止核验](experiments/E98_loss_weight_learning/runs/20261104_dynamic_train500_v3_seed0/stop_verification_20261104.json)。另授权[E103离线Loss对齐研究](experiments/E103_offline_loss_alignment/README.md)：当前仅选定并复制E42/E68/E70/E71/E98泛化增强基线的已完成best及推理源码，先报告模型理由；后续按同1000训练题×5模型建立5000草稿数据库。未启动数据库、后处理或训练，不影响E101/E102。

用户另行授权[E102条件布局图片兴趣探索](experiments/E102_reference_image_generation/README.md)：使用原始面积与reference，近期只研究图片相似程度、小模块像素失真和边界清晰度；不做MIB同形、矩形／矩阵提取、几何修复、后处理或Q评价。源码及五例分辨率观察独立保存，正式长训练尚未启动；不改变E101运行与测试范围。

2026-11-04用户新增并明确授权[三阶段均64k的独立完整运行](experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/README.md)：E42随机64k→E74后训练64k→C5 F64k→冻结→dev500→已登记2000例重复测试，准备完成后自行后台启动。保持原设计与正常早停、无时间/轮数上限；E85教师仅扩展新增48k训练目标。该运行等待前一链路评价结束，五例必要短检通过后串行训练。原交付与所有旧结果保护，不自动按测试改选模型。

用户随后要求补齐[原历史最佳C5的同2000例测试](experiments/E101_final_c5_solver/runs/20261104_original_C5_test2000/README.md)。已核实原C5的raw2000为dev草稿，先前两次测试均为新重训权重。本次固定原`028ea3...`权重、复用历史dev500，不重训、不调参。最新决定为停止64k并优先测试原C5：64k在63561步安全保存，相关进程已退出、后续阶段已阻止，STOP保持，不自动恢复；原C5测试现已启动。

## 当前后台任务

原64k运行的训练、dev500及2000例测试已结束，权重和全部证据独立保留。用户现已明确授权按历史链路新跑完整流程：**E42随机初始化16k → E74后训练32k → 原C5修正器16k → 冻结唯一模型 → dev500 → 已登记test2000重复评价**。

E42、E74和F均按登记的正常早停结束，无时间或轮数上限。原Loss、LR、batch32/FP32/seed0与完整后处理保留；E42历史前2轮渐增几何项、反传G与选择G+occupancy不同，作为复现例外明确登记。E74接新E42唯一best，F沿用原C3暖启动及原E85教师目标，只重算新L草稿。新运行不会覆盖原交付。

2026-11-04已确认三段训练结束：E42早停54轮/best51，E74早停45轮/best43，F早停18轮/best6。dev500曾因隔离目录遗漏官方评分器的上游导入依赖退出，首例结果未保存、测试未读取；现已补齐原依赖并通过导入/哈希短检，保留冻结模型，从评价阶段继续。修复证据见[依赖修复记录](experiments/E101_final_c5_solver/runs/20261103_replay/evaluation_dependency_fix_20261104/README.md)。

上述历史成员规模链路现已完成dev500与2000例评价，[报告已生成](experiments/E101_final_c5_solver/runs/20261103_replay/report.md)。三阶段64k新运行随后通过五例实际更新/教师/完整评分门禁，已进入E42正式训练，确认train_count=64000、至少100步实际更新，时间/轮数上限均为None；[启动核验](experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/launch_verification.json)。

- [完整新运行方案与停止方法](experiments/E101_final_c5_solver/runs/20261103_replay/README.md) · [登记计划](experiments/E101_final_c5_solver/runs/20261103_replay/plan.json)
- [新流水线状态](experiments/E101_final_c5_solver/runs/20261103_replay/state.json) · [dev500日志](experiments/E101_final_c5_solver/runs/20261103_replay/dev500.log) · [test2000日志](experiments/E101_final_c5_solver/runs/20261103_replay/test2000.log)
- [原64k运行状态](experiments/E101_final_c5_solver/runs/20261103_final64k_seed0/state.json) · [原设计说明](experiments/E101_final_c5_solver/DESIGN.md)
- [三阶段64k运行状态](experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/state.json) · [64k登记计划](experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/plan.json) · [64k E42日志](experiments/E101_final_c5_solver/runs/20261104_chain64k_seed0/E42.log)

## 训练后的唯一最终评价

保持原split_v1、v5/v6嵌套成员及固定监测不变。新链路训练结束后冻结唯一分阶段Loss best组合和源码/权重哈希，先完整dev500及同100例reference/L/F/最终图，再按此次明确授权重复评价local_test2000。该2000例已在原E101运行曝光，不再称全新未见测试。不根据Q或测试结果换权重、调系数、改后处理或再次训练；新运行冻结前不读取测试载荷。

质量、硬合法和逐例完整时间分别报告，包含均值/P95/最大值及250ms超限例数。磁盘与冷启动单列，不用批量摊销冒充逐例延迟。公开MIB平均面积派生任务、本地中性时间Q、旧v2测试曝光和开发自适应边界保留，不称正式比赛成绩。

原C5历史dev500：Q=1.012685893，500/500硬合法、零行回退，完整均时204.464ms/P95 521.080ms。它是历史基线，不是新重训或最终测试成绩；原权重在E101的assets/c5_baseline.pt独立保护。原64k导出和新历史链路导出分开保存。

独立环境.conda/envs/iccad2026；数据路径由[protocol/paths.json](protocol/paths.json)解析。[执行协议](docs/protocol.md) · [工作区结构](docs/workspace_structure.md) · [数据说明](docs/datasets.md) · [GPU工程](docs/gpu_training.md)。

E103用户已确认五模型并授权自主研究、后台长任务及4路并行：同1000训练题×5份原始草稿，E98固定计算深度后处理，600/200/200题目隔离的权重拟合与40题局部下降检查；状态见experiments/E103_offline_loss_alignment/runs/20261104_bank1000x5_seed0/state.json。E98不恢复，测试封存。

用户最新明确授权E103使用第一轮拟合的新Loss，从E98数据增强best71续训一个候选：LR从1e-3开始、无下降保护、取消15轮且不设轮数/时间上限，保留正常早停、原16k/固定train-dev500、增强与Dropout。LR四档1e-3→2.5e-4→1e-4→5e-5，epoch0基准，连续3轮停滞降档、最低档3轮停滞早停；best只依据新验证Loss。新run为experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0，训练入口finetune_learned_v1；原E98 STOP、父模型和离线证据保护，不自动后处理Q或读测试。

用户在E103续训32轮正常早停后明确要求验证：唯一新Loss best29已冻结，按原dev500及E98固定计算深度后处理评价；基线best71五题复现后复用已完成500结果，不按Q重选权重，测试封存。结果位于experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500。

E103最新授权：仅使用E98增强best71和当前新Loss best29进行一次权重拟合。原1000训练题保持不变，增强版结果复用；当前版只有已完成dev500，故补齐同1000训练题后处理，不把验证Q拿来拟合。四路后台补齐、固定lambda1仅fit600一次拟合，200/200只报告，结束保存提案即停。入口experiments/E103_offline_loss_alignment/pair_weight_refit，状态runs/20261104_pair_enhanced_current_v1/state.json。


用户2026-11-04另行授权[E104扩容实验](experiments/E104_scaling32k/README.md)：E98泛化增强设计从随机初始化扩大为冻结train32k、128宽×16层、3,463,747参数，原固定Loss/增强/Dropout与正常LR早停保持，无硬时间/轮数上限。准备完成后后台训练，正常结束冻结唯一Loss-best，自动同固定dev500与E98.depth_budget.v1评价并报告，用户已确认不是封存test；不按Q重选或追加候选，不恢复旧E98搜索，不改变E101/E102/E103。状态入口experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0。


E104用户已在准备完成后明确要求“启动”：解除原准备暂停，后台从头训练32k/128×16唯一候选，正常早停后冻结唯一Loss-best并自动同固定dev500评价。原Loss、增强、调度与无硬预算配置不变，不读取封存test，不追加候选。

E103最新研究目标已纠正为跨题目全局Loss–Q相关性：旧排序数据不用于新分析；增强best71/当前best29各负责新1000道不同训练题，共2000独立观察，排除旧1000题，所有草稿/Q重新计算。1400拟合/600检查，样本等权全局非负回归、相关性和图，保留问题间差异，不进行同题排序。入口experiments/E103_offline_loss_alignment/global_loss_q，输出runs/20261104_global_independent2000_seed0；测试封存、不自动训练。

用户新增并明确授权E103/reference075_posttrain小实验：E98原增强best71，position=0.75、shape=0.1875、其他8项原值，不归一化；LR从倒数第二档1e-4至5e-5，无15轮或时间上限，按验证Loss正常早停；后台训练→冻结唯一best→自动E98固定计算深度dev500→报告即停。已有父模型500结果及已通过精确复现桥复用，测试封存。run为experiments/E103_offline_loss_alignment/runs/20261104_reference075_seed0，总体状态pipeline_state.json，日志reference075_posttrain/stdout.log。


E104于2026-11-05获用户明确授权修复数值错误并续训：step20350原未归一化局部消息路径触发单例Loss门禁，登记local-prenorm.v2迁移后继续；模型参数/Adam/顺序/游标/RNG/LR/累计时间保留，固定Loss和门禁不变，新路径重建best与平台基准，旧证据归同run的repair_20261105/before。此后结果包含路径修复，不称纯缩放对照；正常早停后仍单次固定dev500，不读测试。


2026-11-05用户明确授权[E105方案一](experiments/E105_gated_prenorm_placer/README.md)：128×16完整分支pre-norm、四路残差门控与SwiGLU，从头随机seed0；其他数据/增强/Dropout概率/Loss/batch/精度/优化器/LR/早停及完整固定dev500评价与E104保持一致。准备与五例最小短检通过后直接后台训练，正常早停冻结唯一best自动评价并报告即停；仅一个候选，不自动补控制或方案二/三、不读测试。成员/源码/方案对齐见non_architecture_alignment.json，运行runs/20261105_gated128x16_seed0。

用户2026-11-05最新授权独立[E106后处理感知Loss](experiments/E106_postprocessor_aware_loss/README.md)：仅实施与准备数据，不启动代理或布局训练。最新为5000道原train题×5份，共25000份草稿，模型128宽×8层、2,105,420参数；3个成熟模型原稿＋1对局部扰动，按题目3000/1000/1000隔离。使用E98固定计算深度后处理、4个单线程CPU任务建立标签；完成保存数据即停。训练入口由authorization.json显式禁用。日志bank_stdout.log、bank_stderr.log，总状态runs/20261105_bank25000_seed0/state.json。

E106最新授权更新：用户明确要求数据生成完成后自动启动代理训练。后台衔接已登记为25000标签完整生成及fit归一化→原五题GPU更新短检→单个固定Loss代理训练→只读内部check1000→停止。模型/面板/科学参数不变，布局训练仍未授权，dev/test不参与；整体状态见experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/autostart_state.json，训练日志同目录critic_stdout.log。


2026-11-05用户授权[E107几何后训练](experiments/E107_geometry_posttrain/README.md)：继承E105 best58，参考位置/形状降为10%（.1/.025），三个精确并集几何主项＋软约束/线长及局部漂移保护，归一化仅原train500；原32k/增强/Dropout/结构保持，新阶段fresh AdamW，LR1e-4→5e-5、epoch0、无保护/硬预算、正常早停。每批绝对/同题父模型相对Loss爆炸门禁，触发即Adam前拒绝、写STOP并保存退出，不重试或衔接评价。五题短检通过后直接后台启动一个候选，仅正常早停自动固定dev500评价报告即停，测试封存，不影响E106或旧STOP。状态runs/20261105_geometry_ref10_seed0。


E107首轮已因严格相对单例Loss门禁停止：22次接受更新后第23批在Adam前拒绝，单例49.70237超过同题父模型3.766752对应门槛37.667519，数值仍有限。STOP保持、latest/best及失败现场保存，进程退出；未完成一轮、未启动终末dev500，不自动恢复或放宽阈值。见experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/stop_verification.json。
