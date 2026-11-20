# 当前结果

[E157 依次训练S1再S2](../experiments/E157_sequential_s1_s2/README.md)：2026-11-20用户明确改为顺序训练并授权后台启动。冻结E156 best79；先训练S1编码器/共享双层S1两次（L6/L8 .2/.8），冻结own-best后，再新建S2编码器/共享双层S2两次（L10/L12 .2/.8）。完整公开Loss，reference及grouping瓶颈监督为0；仅S2 L12加0.2容差移动项，重叠模块减轻移动惩罚。两阶段各279958可训练参数、36k/batch64、各4h/200轮上限；两阶段冻结/恢复/梯度及完整MIB后处理短检通过后启动。最后唯一S2 best运行新dev500完整MIB评价，不读test。队列与日志以实验内记录为准。

[E156 两次共享S0](../experiments/E156_two_s0_stage/README.md) 已用满原4小时预算，完成80轮及第81轮部分更新，best79 dev Loss0.189882已冻结。未启动S1/S2；父权重现供E157顺序训练使用。

[E155 新形状修正版L0](../experiments/E155_fresh_mib_l0/README.md) 已完成，best51固定train500/dev500 Loss为0.360470/0.360367，父权重SHA已冻结。新36k、MIB形状修正监督；本轮未训练后段、未后处理。

[全新随机MIB 40k数据准备](../datasets/random_mib40k_v1/README.md)：2026-11-19用户授权忽略历史重叠、重新随机划分与抽样；先扫描超过40k候选，MIB兼容筛选后补齐36k/2k/2k。新train/dev监督与公开输入分别处理、生成38k增强静态缓存和分层随机监测500；test仅公开规格兼容检查，标签继续封存。启动前短检通过，后台扫描/缓存已启动，不训练模型，状态以数据目录为准。

[E154 移除瓶颈项后重训 L0](../experiments/E154_l0_no_bottleneck/README.md)：已完成53轮正常早停，own-Loss-best为第50轮，原固定train500 Loss0.372355/dev500 Loss0.368000，best已冻结；没有启动后段或后处理。新选[MIB结构挑战dev500](../datasets/mib_dev500_challenge_v1/report.md)及独立CPU缓存已完成，按36k的N分布分配、公开约束复杂度择优，未改变E154原监测和选模结果。后续瓶颈监督仅S0/L4系数0.2，L0/L8/L12为0。

E153两阶段及完整MIB dev500已完成：最终500/500硬合法、MIB全满足，grouping453/500（90.6%），混合Q1.043412；未达到95%目标。[Loss与后处理审计](../experiments/E153_mib_role_topology/runs/loss_postprocessing_audit_v1/report.md)记录了总Loss口径变化及9个外框异常案例。

[E153 MIB与增强题目编码](../experiments/E153_mib_role_topology/README.md) 已按用户要求完成一个scratch方案的准备与检查：原MIB32k/train500/dev500不改划分；先处理数据，L0及所有后层使用增强条件，S0 reference为0.75并加入分组/成员重排。先训L0再训后段，无历史权重迁移。数据处理、全数据尺寸压力检查、两阶段梯度/恢复检查和完整MIB后处理短检均通过；用户已于2026-11-18授权启动，后台依次训练L0与后段，完成后自动原dev500完整MIB评价；实时状态见实验内记录。

[E152 只微调 S2 的 A/B 对照](../experiments/E152_s2_local_repair/README.md) v2 两组已完成原36k、各10轮训练及原dev500。A保留reference（best10），B去掉（best8），最终均500/500硬合法，混合Q为1.008752/1.009692，软约束全满足425/419题，均无外框超过reference1.2倍的题。[统计与逐阶段展示](../experiments/E152_s2_local_repair/runs/review_dev500_v1/report.md) 按本轮用户请求生成；旧版结果已作废。

[E151 固定 E149 初稿并重训后段](../experiments/E151_frozen_role_initial/README.md) 于 2026-11-18 按用户要求安全停训，完成 45 轮和 25508 次更新，保留 best45 与 latest。STOP 已保留，终末 dev500 阶段审计尚未运行。

[E150 新初稿与原后段联合训练](../experiments/E150_role_joint/README.md)：2026-11-18用户授权准备通过后后台启动：E149新角色初稿＋E148原12层后续结构，681141参数全部scratch seed0联合训练。L0/L4/L8辅助各1/15、L12为.8；L0和L4去直接overlap，L8/L12完整。只传布局跨双层单元，初稿也直接监督。原36k/batch64/4h，基础及E149角色缓存复用；只记录Loss，无后处理/Q/test。准备检查已通过，状态以实验内记录为准。

[E149 结构角色初稿独立训练](../experiments/E149_role_initial/README.md)：2026-11-17用户授权准备通过后后台启动，只训练带结构角色编码的初稿生成器。两层width64、完整引用端口token、多跳/锚点角色和静态关系bias，共107739参数；无S0/S1/S2后段。初稿唯一输出直接监督，完整去掉overlap组、其余原Loss系数保持。原36k/batch64、4h/200epoch、原train500/dev500，基础缓存复用，仅新建公开role sidecar；只记录Loss，无后处理/Q/test。准备检查已通过，实时状态以实验内记录为准。

[E148 联合双层功能网络与形变反馈](../experiments/E148_shared_functional_shape/README.md)：2026-11-17按用户修订完成准备并后台启动。三套双层单元按S0/S0/S1/S1/S2/S2共享调用，跨单元仅传布局，第一子层读取纯题目条件、第二子层不重复注入。保留E142原49维反馈、无空白统计，新增16维等面积形变响应；678941参数联合scratch seed0。仅4/8/12层监督，权重.1/.1/.8，4去直接overlap、8/12完整；原36k/batch64、整个候选4h上限及自动原dev500。准备检查已通过；实时状态以实验内记录为准。

[E147 监督层后清除隐藏传递](../experiments/E147_supervision_hidden_reset/README.md)：2026-11-17用户授权准备后后台启动。仅S2→S3、S6→S7、S8→S9、S10→S11重建隐藏输入，其余层保留隐藏传递，布局及其梯度始终传递。E142原49维反馈、pair bias/ψ、895123参数、batch64/36k、E139监督、scratch seed0及原4h/早停设置保持。训练后自动原固定dev500和Q日志；实时状态以实验内记录为准。


[E144/E145/E146 三组E142对照](../experiments/E144_local_blank/README.md)：2026-11-16用户授权准备后后台依次scratch训练，每组自动原固定dev500。E144为截断至对应边长的脱离距离＋0.25边长新条带空白比例（894867参数）；E145仅移除ψ关系value、保留pair bias（869779）；E146层间仅传布局，每层从公开静态属性重构输入，不传前层或S0隐藏状态（895123）。均batch64、原36k、E139监督和原优化/早停/4h上限；准备检查已通过。整体实时状态在E144 `background/suite_state.json`，总日志 `background/suite.stdout.log`；Q逐组写日志，错误/STOP阻断后续。


核对日期：2026-11-16。这里是静态摘要，实时运行以实验内状态和实际进程为准。

[E143 四方向可移动距离实验](../experiments/E143_directional_travel/README.md)：2026-11-16用户授权准备通过后后台scratch训练，仅改E142方向反馈：阈值0.1、连续首次越阈距离、到边界/对应边长双上限，保留0.25s响应、删除1.0s响应。45维动态输入、894867参数，其他E142设置及E139监督保持；准备检查通过，自动原固定dev500和Q日志。实时状态看实验内记录。

[E142 模块动态需求与关系value实验](../experiments/E142_dual_geometry/README.md)：2026-11-16用户授权准备通过后后台scratch训练，batch64、895123参数、冻结36k；监督/Loss完整复用E139（S2/S6去overlap，S8/S10/S12完整，.05/.05/.05/.05/.8），不加入阻隔Loss。49维动态摘要经需求编码进入QKV，原18维pair bias保留，新增接收者相关pair value。启动前检查已通过，随后自动原固定dev500/原后处理/独立硬检查，Q摘要进入日志。实时状态以实验内记录为准，暂无新结果。

[E141 三组路径阻隔/逐层监督实验](../experiments/E141_path_supervision/README.md)：2026-11-16用户授权准备并后台依次训练，batch64、scratch seed0、冻结36k；每组成功后自动原dev500后处理/独立硬检查。实验一S3/6/9/12完整监督加阻隔项；二为S1–S12逐层监督（仅S3/6/9/12含overlap）；三为组合。S0不监督、末层.8、辅助共.2、不沿用G/B×4。三组准备检查通过，后台顺序队列已于2026-11-16启动，首组开始加载兼容现有训练缓存；实时状态以实验内 `preflight.json` / `queue_state.json` 为准，暂无新结果。

[E140前段G/B四倍结果](../experiments/E140_frontend_gb4/result_review.md) 已完成训练和原dev500后处理：56轮正常早停、best53，训练8690.6秒；混合Q1.039537，最终500/500硬合法、全部软约束满足378/500（E139为384），分组违规106（83）、贴边82（54），完整均时/P95为197.1/481.8ms。共同末层完整Loss .341736虽低于E139 .354014，但最终指标退化，dev358的Q1.088850→4.311288。唯一变化为S2/S6原加权G/B各×4，后面三层与其余配置保持；参考关系图指标未加入。本轮未追加拓扑审计，不据最终结果推断反馈机制的因果作用，不追加运行。

[E139延后重叠监督](../experiments/E139_delayed_overlap/README.md) 的训练与原dev500后处理均complete：45轮正常早停、best42、训练约6968秒，混合Q1.016845，最终500/500硬合法、草稿0/500，完整均时/P95为173.3/406.4ms。S2/S6无直接碰撞项，S8/S10/S12完整Loss，前四项各.05、末层.8，原E119/36k/scratch seed0。

[E139与E134每三层配对审计](../experiments/E139_delayed_overlap/runs/e119_delayed_overlap_seed0/comparative_audit_v1/findings.md) 已完成，同dev500/299源分片。最终grouping违规104→83、贴边47→54；混合Q差−.001779中grouping公式记账贡献−.002246。组外框和异组侵入略好，但接触瓶颈.129623→.137264、固定1170条受阻边严格修复352→353、替代路径修复452→429，局部方向/K6略低。S3后完整换侧2→7次、仅3题。Q差的分片bootstrap区间跨0，收益受个别尾部题影响；不能据此宣称可靠拓扑突破。只补冻结推理/统计，原权重、源码和后处理未改，无新训练/后处理/test。

[E138拓扑重规划三组](../experiments/E138_topology_rewrite/outcome_20261115.md) 的训练、原dev500后处理与结构审计均已完成，队列complete。A/B/C混合Q为1.028175/1.034249/1.045416，共同末层新Loss为.362372/.372267/.392470，均差于复用E134每三层best63的1.018624/.334363；最终均500/500硬合法。用户2026-11-15宣告本轮隐空间重规划路线失败，不自动追加候选或恢复历史任务。详见原`comparison.md/json`及`evaluation_log.md`；原冻结源码与权重不改。

[E134非监督层结构审计](../experiments/E134_supervision_interval/runs/e119_stride3_seed0/unsupervised_audit_v1/findings.md) 已完成：S3之后500题仅2次完整换侧，但局部交错、阻隔解除和空隙修复真实存在；不能把Loss反弹本身当作突破坏拓扑的证据。原权重与后处理未改。

[E134每三层基线的逐层分析](../experiments/E134_supervision_interval/runs/e119_stride3_seed0/layer_analysis_v1/findings.md) 已完成：固定best63，同一dev500/299源分片，原S0–S12、已有15例和原末层逐位核对。S1–S3完成S0→S12局部方向/K6新增净增益约94.5%/91.8%；终态分离方向持续≥95%一致的首次中位层为S11。S3→S12的Loss下降约63.7%来自碰撞项。S0隐状态的非P交叉验证线性探针优于S0实际布局；原模型未更新、梯度无优化器、无新后处理或test读取。宏观关系、细分离定型与任务可拼合性分开判断，不据此证明拓扑必然更难或前端容量不足。

[E137标准Transformer256](../experiments/E137_standard_transformer256/README.md) 已于2026-11-14约21:23（台北）按用户要求终止，队列`cancelled_by_user`，训练worker为`user_stop`且无错误。保留107完整轮/120375接受步，best104的固定dev Loss为1.003466667；best/latest和完整恢复现场均保留。训练PID23160、队列PID28596已退出，GPU占用回落；授权关闭，STOP/QUEUE_STOP保留，后续dev500后处理及地址诊断已阻断，postprocessing_calls=0/test_read=false，不自行恢复。记录在E137 `background/termination_verified.json`。本模型为36k scratch seed0、可学习编码＋单层标准4头Transformer，d256/FFN1024、1,335,363参数、最终完整Loss；尚无最终Q结果，也不称训练充分拟合。

[E136 D：S0 F＋线性增权](../experiments/E136_frontend_F_linear/README.md) 已正常早停并完成原dev500后处理，49轮/best46，混合Q1.021426（平均1.023501、加权1.019352），最终500/500硬合法，完整均时/P95为202.3/497.2ms。共同末层完整新Loss为.353143，own-Loss为.367922。原件在E136 `comparison.json` 与 `runs/e119_D_seed0/evaluation_no_mib_dev500/`。本轮按用户要求仅补充Reference／五组S12草稿／原后处理的同题Loss拆项与几何统计，位置为E136 `runs/statistics_dev500_v1/`，不重训、不重跑后处理、不作原因分析、不读test。

[E135 S0监督 A/B/C](../experiments/E135_initial_supervision/README.md) 已全部正常早停并自动完成原dev500后处理，best53/42/42，混合Q **1.037931/1.031578/1.019951**，均500/500最终硬合法、草稿0/500。共同末层完整新Loss为 .344981/.349986/.357069；已有E134每三层best63为 .334363、混合Q1.018624。C在三组中Q最好，接近但未优于已有对照；A/B最坏Q为3.273556/3.176095，均对应dev358。完整[对比报告](../experiments/E135_initial_supervision/runs/results_review.md)保留不同own-Loss、三种Q、尾部与耗时。各run的 `visualization_dev15_loss/` 已存相同普通10例＋各自最坏5例的S0–S12与逐层Loss；C的S0单列实际F。展示仍使用统一入口，端口8784/8785/8786。

用户最新判断：历史E119单seed的Q≈1.02不作为可靠性能标杆。E134的E119每三层/每两层组已训练并完成原dev500，best63/51，混合Q1.018624/1.022760，均500/500最终硬合法。E120每三层组在38轮/43543接受步触发数值门禁退出；用户决定放弃继续研究E120，另一组未启动。队列原needs_attention现场保留，STOP/QUEUE_STOP及授权关闭，详见 `experiments/E134_supervision_interval/cancellation_e120_20261113/`。两组E119的逐层Loss展示使用各自run下的 `visualization_dev15_loss/`。

## 原固定dev500，原预算后处理

| 模型 | best epoch | 混合Q↓ | 最终硬合法 | 草稿硬合法 | 完整均时/P95 ms |
|---|---:|---:|---:|---:|---:|
| E119 global2（研究基线） | 55 | 1.021568 | 500/500 | 0/500 | 193.4/497.8 |
| E120 Node12 | 79 | 1.039004 | 500/500 | 0/500 | 215.6/571.7 |
| E119 public_priority_v1 | 56 | 1.033645 | 500/500 | 0/500 | 207.3/532.2 |
| E133 FM（独立评价） | 239 | 1.157321 | 500/500 | 0/500 | 206.4/499.3 |

Q为原面积/无MIB本地研究分数；混合Q为加权Q和平均Q的平均，越低越好。E133训练16k、E119/120为36k，目标/参数不同；单seed、长期复用dev和历史运行环境不支持纯架构因果比较。完整耗时尾部仍存在，不能称逐例250ms达标。

## 状态与原件

| 实验 | 已核对状态 | 主要结果位置 |
|---|---|---|
| E119 / E120 | 队列complete | 各自runs下evaluation_no_mib_dev500/results.json |
| E119 public_priority_v1 | 队列complete；best56，新Loss未改善最终Q | variants/public_priority_v1/runs/global2_seed0/ |
| E131 64×4 | user_stop；full在93轮处停，pure未启动 | E131/queue_state.json及runs；STOP保留 |
| E132 96×6 | complete；full153轮/best145/Loss0.403183，pure144轮/best136/Loss0.775958 | E132/runs/full_seed0与pure_seed0/results.json；无后处理/Q |
| E133 FM | 247轮正常早停/best239；原queue保留needs_attention | runs/fm_seed0；独立成功评价evaluation_quality_20261113/ |

E133原自动评价因重复dict字段在0例退出；本次独立评价的500例成功结果不回写原失败队列、不重训、不修改权重、不清零原8h预算。其[完整报告](../experiments/E133_flow_matching/evaluation_quality_20261113/report.md)及逐例坐标均保留。冻结权重SHA256为 `1510d786df358ad3e4390596b948dd2c64289945b7f4639ef1ed9acef80d352f`。

E119 best55已授权的独立test500也已完成：混合Q1.039876，最终500/500硬合法。结果位于 `experiments/E119_global_initial_generator/evaluation_test500_20261109/recovery1_20261109/output/results.json`。该500例已曝光，其余1500例继续封存。

竞赛交付保持归档E101原C5＋完整后处理；历史E119研究身份见 `protocol/research_baseline_v3.json`。E135为新授权实验，不恢复旧任务、不读取test载荷，详见页首。

布局展示统一入口：[visualization/README](../visualization/README.md)。根output已删除；旧长期文档与可视化原件见 `history/20261113_workspace_cleanup/`。
