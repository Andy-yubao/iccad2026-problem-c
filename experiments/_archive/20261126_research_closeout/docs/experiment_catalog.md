# 实验位置索引

这是迁移前的实验索引。现行的 E1–E170 目录与版本位置见[路径对照](../../path-map.json)，实验方法与结论见[实验综述](../../../../deliverables/实验流程记录_E1-E170.md)。

[E170 五阶段布局细化](../../../E170_three_block_stages/README.md)直接复用预训练 L0 为 S0，后续四阶段各三层 attention，逐阶段本地训练设计。

[E169 Ref080](../../../E169_s2_progressive_unfreeze/README.md)在 E168 上训练 S2。L12 reference 倍率 0.80，保留分组和贴边代理以及两项计算优化。[dev2000 终末结果](../../../E169_s2_progressive_unfreeze/evaluation/output/results.json)。

[E168 S1](../../../E168_s1_progressive_unfreeze/README.md)从 E167 best57 初始化，分阶段冻结和解冻。[S1 对照](../../../E168_s1_progressive_unfreeze/E168_E165_S1_comparison.json)。

[E167 S0](../../../E167_s0_no_overlap_feedback/README.md)从 E165 独立 L0 出发，关闭 S0 的重叠几何反馈和直接重叠项。[S0 对照](../../../E167_s0_no_overlap_feedback/comparison.json)。

[软约束后处理及坐标等价加速](../research/postprocess_tail/README.md)记录 E165 dev2000 的质量与耗时对照。

[E166 联合后段的软约束监督](../../../E166_soft_count_joint/README.md)复用 E165 L0 best67，完整后段联合训练，前 20 轮冻结 L0。L12 reference 系数为 0.25，L8/L12 加入分组与贴边代理项，以原 no-MIB dev2000 做终末检查。

[E165 三层初稿与后段联合微调](../../../E165_joint_l0/README.md)使用完整原 train36k、宽 64 × 3 静态关系 L0 和 E151 后段。L0 best 单独备份，后段先冻结 L0 20 轮，解冻后端到端微调，自动运行原 no-MIB dev2000。

[E164 两层宽 64 静态关系 L0](../../../E164_static_relation_l0/README.md)使用原 no-MIB 完整 36k 从头训练，保留原 MST 分组接触及 0.25 倍原重叠项。

[E163 · E151 的 S2 首次调用半步](../../../E163_e151_s2_half_step/README.md)。冻结 E151 best45，原 no-MIB dev2000 背景对照，附独立草稿延迟和后段 CUDA Graph 重放测速。

[E162逐阶段半步/全步重复调用](../../../E162_repeat_half_residual/README.md)：固定E161 S2 best21，不训练。原dev500 Loss对照已完成，S0/S1/S2降低2.22%/3.87%/3.40%；后台原no-MIB dev2000进程PID20760已创建，[实时状态](../../../E162_repeat_half_residual/evaluation/output/state.json)。E161原dev2000已完成，混合Q1.031281153、2000/2000硬合法。以下为历史过程。

E161本地dev2000已按用户2026-11-23指令正式启动，使用回传冻结的S2 best21，不训练、不改原预算、不读取test。实时进度见[状态](../../../E161_final_no_mib/_snapshots/kaggle_transfer/local_dev2000_v1/output/state.json)，[评价入口](../../../E161_final_no_mib/_snapshots/kaggle_transfer/local_dev2000_v1/README.md)。以下为历史过程。

2026-11-23台北16:43：已通过Kaggle API核实E161四阶段训练完成并回传全部70个远端文件（97962747字节）。四阶段best均通过冻结SHA256与本地CPU读取检查；S1完成43轮/预算结束、best40，S2完成26轮/正常早停、best21（own-Loss0.247930124）。S1/S2 latest含终末模型/Adam/RNG/游标/调度/投影及累计预算。远端因打包漏带evaluation/budget.json在dev2000入口导入阶段出错，2000例未开始、没有Q结果；原evaluation/plan.json也未入包，两份配置已保留供本地评价。原本地暂停运行未覆盖。远端版本1已ERROR终止，额度接口两次确认GPU time_reserved=0秒，无需取消已终止会话；未删除Notebook/数据集/结果。后续在本地继续，本次未启动本地评价。[回传与检查记录](../../../E161_final_no_mib/_snapshots/kaggle_transfer/downloaded/20261123T084205Z/RETURN_TO_LOCAL.md)。以下为历史过程。

2026-11-23台北10:14：E161 Kaggle私有GPU任务已成功提交（kernel146771988，version1），最近实际返回QUEUED。私有Dataset version1已READY，两个文件大小均匹配；认证账户andyyubao已核验。本轮尚未确认新的训练更新，不能写为已开始训练。任务将从S1 step7568接续，再训练S2并自动原no-MIB dev2000；保留累计预算，本地E161保持暂停，不重复提交。[任务](https://www.kaggle.com/code/andyyubao/iccad-e161-s1-resume-20261123)；[提交凭据](../../../E161_final_no_mib/_snapshots/kaggle_transfer/submission.json)。以下为历史过程。

[E161 Kaggle迁移](../../../E161_final_no_mib/_snapshots/kaggle_transfer/README.md)：准备私有上传包、S1暂停状态的跨GPU身份迁移及S2/原dev2000后续任务；当前宿主网络权限阻断，未提交。原E161保持暂停，不另建模型候选或seed。

[E161](../../../E161_final_no_mib/README.md)已按用户要求准备独立新scratch运行，当前`prepared`、由用户亲自启动。[启动脚本](../../../E161_final_no_mib/launch_user_scratch.ps1)；[旧运行和预算归档](../../../E161_final_no_mib/_snapshots/before_user_scratch_20261122T160405Z/archive_manifest.json)。原架构、数据和终末dev2000不变；以下后台启动描述均为旧运行过程。

[E161 最终无MIB四阶段架构](../../../E161_final_no_mib/README.md)已准备、未启动训练/评价（2026-11-22）。E151 best45的[原reference dev2000评价](../../../E151_frozen_role_initial/runs/evaluation_reference_dev2000_v1/README.md)已按本轮授权创建后台进程，未执行启动验证。最终数据恢复原random_reference40k，当前面板见[no-MIB dev2000](../protocol/final_evaluation_no_mib_dev2000_v1.json)。

[2026-11-22收尾设计与结果核对](final_architecture_20261122.md)：E160已完成best63、dev Loss0.335786179；E157新MIB dev500混合Q1.014650315、grouping461/500；E151 best45历史no-MIB dev500混合Q1.008437267。新四阶段架构与[完整dev2000成员](../protocol/final_evaluation_dev2000_v1.json)已登记，未启动新训练或评价。以下历史启动描述保留为过程记录，最新完成状态见[当前结果](current_status.md)。

[E160 静态关系消息初稿](../../../E160_static_relation_message/README.md) 已按用户授权完成准备并创建后台正式训练进程（2026-11-22）；按AGENTS未执行启动验证，运行状态以实验记录为准。在E159三个attention块中加入秩8的接收者相关静态关系消息，不增加分跳聚合；width128/四头、数据/MIB/Loss、36k/batch64/seed0、4h/200轮预算保持。仅L0，完成后冻结own-best并自动比较E159保存的总Loss与分项。未执行防御性测试、试训或启动验证，无后段/后处理/test；启动记录和日志见实验background目录。

[E159 加宽三层 RoleInitial](../../../E159_role_initial128/README.md) 已完成50轮正常早停，best47，固定train500/dev500 Loss0.348669/0.347660，耗时79.9分钟。相对E158 best58的dev总Loss降低8.74%，482/500题改善；线长/紧凑分项略升。[完整比较](../../../E159_role_initial128/comparison_e158/report.md)。同题对照10例和E159 Loss最差10例已保存，端口8829/8830。仅L0，无后处理或test。

[E158 三层 RoleInitial 初稿](../../../E158_role_initial3/README.md) 已完成61轮正常早停，34343次更新，耗时89.6分钟，冻结best58；固定train500/dev500 Loss为0.381521/0.380959（含reference1.0和原重叠组×0.25）。[分项Loss、曲线与E155共同口径对照](../../../E158_role_initial3/runs/review_results_v1/report.md) 已保存；随机dev20已按统一入口展示，端口8828。仅L0，无后段、后处理或test。

[E157 依次训练S1再S2](../../../E157_sequential_s1_s2/README.md)：2026-11-20用户明确改为顺序训练并授权后台启动。冻结E156 best79；先训练S1编码器/共享双层S1两次（L6/L8 .2/.8），冻结own-best后，再新建S2编码器/共享双层S2两次（L10/L12 .2/.8）。完整公开Loss，reference及grouping瓶颈监督为0；仅S2 L12加0.2容差移动项，重叠模块减轻移动惩罚。两阶段各279958可训练参数、36k/batch64、各4h/200轮上限；两阶段冻结/恢复/梯度及完整MIB后处理短检通过后启动。最后唯一S2 best运行新dev500完整MIB评价，不读test。队列与日志以实验内记录为准。

[E156 两次共享S0](../../../E156_two_s0_stage/README.md) 已用满原4小时预算，完成80轮及第81轮部分更新，best79 dev Loss0.189882已冻结。未启动S1/S2；父权重现供E157顺序训练使用。

[E155 新形状修正版L0](../../../E155_fresh_mib_l0/README.md) 已完成，best51固定train500/dev500 Loss为0.360470/0.360367，父权重SHA已冻结。新36k、MIB形状修正监督；本轮未训练后段、未后处理。

[E154 移除瓶颈项后重训 L0](../../../E154_l0_no_bottleneck/README.md)：已完成53轮正常早停，own-Loss-best为第50轮，原固定train500 Loss0.372355/dev500 Loss0.368000，best已冻结；没有启动后段或后处理。新选[MIB结构挑战dev500](../datasets/mib_dev500_challenge_v1/report.md)及独立CPU缓存已完成，按36k的N分布分配、公开约束复杂度择优，未改变E154原监测和选模结果。后续瓶颈监督仅S0/L4系数0.2，L0/L8/L12为0。

E153两阶段及完整MIB dev500已完成：最终500/500硬合法、MIB全满足，grouping453/500（90.6%），混合Q1.043412；未达到95%目标。[Loss与后处理审计](../../../E153_mib_role_topology/runs/loss_postprocessing_audit_v1/report.md)记录了总Loss口径变化及9个外框异常案例。

[E153 MIB与增强题目编码](../../../E153_mib_role_topology/README.md) 已按用户要求完成一个scratch方案的准备与检查：原MIB32k/train500/dev500不改划分；先处理数据，L0及所有后层使用增强条件，S0 reference为0.75并加入分组/成员重排。先训L0再训后段，无历史权重迁移。数据处理、全数据尺寸压力检查、两阶段梯度/恢复检查和完整MIB后处理短检均通过；用户已于2026-11-18授权启动，后台依次训练L0与后段，完成后自动原dev500完整MIB评价；实时状态见实验内记录。

[E152 只微调 S2 的 A/B 对照](../../../E152_s2_local_repair/README.md) v2 两组已完成原36k、各10轮训练及原dev500。A保留reference（best10），B去掉（best8），最终均500/500硬合法，混合Q为1.008752/1.009692，软约束全满足425/419题，均无外框超过reference1.2倍的题。[统计与逐阶段展示](../../../E152_s2_local_repair/runs/review_dev500_v1/report.md) 按本轮用户请求生成；旧版结果已作废。

[E151 固定 E149 初稿并重训后段](../../../E151_frozen_role_initial/README.md) 含冻结初始化、后段联合训练和 dev500 S1 阶段审计。训练总目标为 L4/L8/L12 的 0.1/0.1/0.8。

[E150 新初稿与原后段联合训练](../../../E150_role_joint/README.md)：2026-11-18用户授权准备通过后后台启动：E149新角色初稿＋E148原12层后续结构，681141参数全部scratch seed0联合训练。L0/L4/L8辅助各1/15、L12为.8；L0和L4去直接overlap，L8/L12完整。只传布局跨双层单元，初稿也直接监督。原36k/batch64/4h，基础及E149角色缓存复用；只记录Loss，无后处理/Q/test。准备检查已通过，状态以实验内记录为准。

[E149 结构角色初稿独立训练](../../../E149_role_initial/README.md)：2026-11-17用户授权准备通过后后台启动，只训练带结构角色编码的初稿生成器。两层width64、完整引用端口token、多跳/锚点角色和静态关系bias，共107739参数；无S0/S1/S2后段。初稿唯一输出直接监督，完整去掉overlap组、其余原Loss系数保持。原36k/batch64、4h/200epoch、原train500/dev500，基础缓存复用，仅新建公开role sidecar；只记录Loss，无后处理/Q/test。准备检查已通过，实时状态以实验内记录为准。

[E148 联合双层功能网络与形变反馈](../../../E148_shared_functional_shape/README.md)：2026-11-17按用户修订完成准备并后台启动。三套双层单元按S0/S0/S1/S1/S2/S2共享调用，跨单元仅传布局，第一子层读取纯题目条件、第二子层不重复注入。保留E142原49维反馈、无空白统计，新增16维等面积形变响应；678941参数联合scratch seed0。仅4/8/12层监督，权重.1/.1/.8，4去直接overlap、8/12完整；原36k/batch64、整个候选4h上限及自动原dev500。准备检查已通过；实时状态以实验内记录为准。

[E147 监督层后清除隐藏传递](../../../E147_supervision_hidden_reset/README.md)：2026-11-17用户授权准备后后台启动。仅S2→S3、S6→S7、S8→S9、S10→S11重建隐藏输入，其余层保留隐藏传递，布局及其梯度始终传递。E142原49维反馈、pair bias/ψ、895123参数、batch64/36k、E139监督、scratch seed0及原4h/早停设置保持。训练后自动原固定dev500和Q日志；实时状态以实验内记录为准。


[E144/E145/E146 三组E142对照](../../../E144_local_blank/README.md)：2026-11-16用户授权准备后后台依次scratch训练，每组自动原固定dev500。E144为截断至对应边长的脱离距离＋0.25边长新条带空白比例（894867参数）；E145仅移除ψ关系value、保留pair bias（869779）；E146层间仅传布局，每层从公开静态属性重构输入，不传前层或S0隐藏状态（895123）。均batch64、原36k、E139监督和原优化/早停/4h上限；准备检查已通过。整体实时状态在E144 `background/suite_state.json`，总日志 `background/suite.stdout.log`；Q逐组写日志，错误/STOP阻断后续。


2026-11-16核对。目录存在不代表可以启动；状态和权重身份看实验自身记录。

## 现有实验

[E143 四方向可移动距离实验](../../../E143_directional_travel/README.md)：2026-11-16用户授权准备通过后后台scratch训练，仅改E142方向反馈：阈值0.1、连续首次越阈距离、到边界/对应边长双上限，保留0.25s响应、删除1.0s响应。45维动态输入、894867参数，其他E142设置及E139监督保持；准备检查通过，自动原固定dev500和Q日志。实时状态看实验内记录。

[E142 模块动态需求与关系value实验](../../../E142_dual_geometry/README.md)：2026-11-16用户授权准备通过后后台scratch训练，batch64、895123参数、冻结36k；监督/Loss完整复用E139（S2/S6去overlap，S8/S10/S12完整，.05/.05/.05/.05/.8），不加入阻隔Loss。49维动态摘要经需求编码进入QKV，原18维pair bias保留，新增接收者相关pair value。启动前检查已通过，随后自动原固定dev500/原后处理/独立硬检查，Q摘要进入日志。实时状态以实验内记录为准，暂无新结果。

[E141 三组路径阻隔/逐层监督实验](../../../E141_path_supervision/README.md)：2026-11-16用户授权准备并后台依次训练，batch64、scratch seed0、冻结36k；每组成功后自动原dev500后处理/独立硬检查。实验一S3/6/9/12完整监督加阻隔项；二为S1–S12逐层监督（仅S3/6/9/12含overlap）；三为组合。S0不监督、末层.8、辅助共.2、不沿用G/B×4。准备与实时状态以实验内 `preflight.json` / `queue_state.json` 为准；暂无新结果。

最新：[E140_frontend_gb4结果](../../../E140_frontend_gb4/result_review.md)，56轮正常早停/best53、原dev500后处理complete。混合Q1.039537，全部软约束满足378/500，分组/贴边违规106/82，差于E139；最终500/500硬合法。相对E139仅S2/S6原加权G/B各×4，后面三层及其余设置保持，无新拓扑指标；不追加运行。

[E139_delayed_overlap](../../../E139_delayed_overlap/README.md)，45轮正常早停、best42，原dev500后处理complete，混合Q1.016845、最终500/500硬合法。[与E134每三层的其他指标及拓扑审计](../../../E139_delayed_overlap/runs/e119_delayed_overlap_seed0/comparative_audit_v1/findings.md) 已完成：最终grouping违规104→83，但严格固定坏关系修复无整体优势，尚无可靠拓扑突破证据。原E119、36k、scratch seed0；S2/S6无直接碰撞项、S8/S10/S12完整Loss，前四项各.05、S12为.8。原现场保留，不追加运行。

[E138_topology_rewrite](../../../E138_topology_rewrite/outcome_20261115.md)，A/B/C训练、原dev500后处理和结构诊断均complete；混合Q1.028175/1.034249/1.045416，均差于E134对照1.018624。用户于2026-11-15宣告本轮隐空间重规划路线失败，原现场保留，不追加运行。

[E137_standard_transformer256](../../../E137_standard_transformer256/README.md) 已按用户要求终止于107完整轮/120375接受步；best104及latest保留，授权关闭、STOP保留，未执行后处理/Q和地址诊断，不自行恢复。

| 实验 | 实际目录 |
|---|---|
| E119 | [E119_global_initial_generator](../../../E119_global_initial_generator/) |
| E120 | [E120_node12_collision](../../../E120_node12_collision/) |
| E131 | [E131_relation_input](../../../E131_relation_input/) |
| E132 | [E132_relation_input_96x6](../../../E132_relation_input_96x6/) |
| E133 | [E133_flow_matching](../../../E133_flow_matching/) |

新增入口：[E134_supervision_interval](../../../E134_supervision_interval/README.md)，E119间隔2/3已完成，E120放弃并保留停止现场；[E135_initial_supervision](../../../E135_initial_supervision/README.md)，E119 S0监督 A/B/C与自动原dev500已全部完成，结果对比在 `runs/results_review.md`，S0–S12逐层Loss展示在各run的 `visualization_dev15_loss/`。

## 近期归档

现行新增：[E136_frontend_F_linear](../../../E136_frontend_F_linear/README.md)，D（S0 F＋线性增权），准备通过后按用户授权后台启动；自动后处理并复用既有对照，运行状态以实验内记录为准。

下列29个实验统一位于 [history/20261111_experiments_consolidation/experiments/](../../20261111_experiments_consolidation/experiments/)，完整迁移/哈希见其 [归档说明](../../20261111_experiments_consolidation/README.md)。

| 实验 | 该归档内目录 |
|---|---|
| E98 | `E98_loss_weight_learning/` |
| E101 | `E101_final_c5_solver/` |
| E102 | `E102_reference_image_generation/` |
| E103 | `E103_offline_loss_alignment/` |
| E104 | `E104_scaling32k/` |
| E105 | `E105_gated_prenorm_placer/` |
| E106 | `E106_postprocessor_aware_loss/` |
| E107 | `E107_geometry_posttrain/` |
| E108 | `E108_random_reference40k/` |
| E109 | `E109_deep_supervision_shared/` |
| E110 | `E110_deep_supervision_heads/` |
| E111 | `E111_progressive_depth/` |
| E112 | `E112_deep_geometry_budget/` |
| E113 | `E113_deep_head_budget/` |
| E114 | `E114_supervision_placement/` |
| E115 | `E115_first_head_supervision/` |
| E116 | `E116_first_draft_comparison/` |
| E117 | `E117_node26/` |
| E118 | `E118_first12_supervision/` |
| E121 | `E121_residual_update/` |
| E122 | `E122_global2_scaling16/` |
| E123 | `E123_node12_scaling16/` |
| E124 | `E124_residual_scaling16/` |
| E125 | `E125_global2_ref04/` |
| E126 | `E126_global2_refsplit/` |
| E127 | `E127_node12_ref04/` |
| E128 | `E128_node12_refsplit/` |
| E129 | `E129_global2_s0ref/` |
| E130 | `E130_node12_s0ref/` |

竞赛交付是E101原C5＋完整后处理，E119是研究基线。E101部分隔离运行根超过MAX_PATH，读取使用Windows长路径。更早E1–E97按归档事件导航见 [history/README](../../README.md)，不把旧路径/授权当现行入口。

2026-11-13的旧文档与全部旧可视化见 [本次清理归档](../../20261113_workspace_cleanup/README.md)。现行展示只调用 `visualization`；历史实验私有显示脚本不作为第二套默认。
