# ICCAD 2026 Problem C

[E166 联合后段的软约束监督](experiments/E166_soft_count_joint/README.md)已完成，原 no-MIB dev2000 混合 Q 为 1.011756，硬合法 2000/2000。复用 E165 L0 best67，降低 L12 reference 并在 L8/L12 加入分组和贴边代理项。草稿贴边改善，最终混合 Q 仍高于 E165。[结果](experiments/E166_soft_count_joint/evaluation/output/results.json)

[E165 三层初稿与后段联合微调](experiments/E165_joint_l0/README.md)已完成，原 no-MIB dev2000 混合 Q 为 1.008717，硬合法 2000/2000。完整原 train36k 为当前默认，L0 best67 已在本地单独备份。[结果](experiments/E165_joint_l0/evaluation/output/results.json) · [当前数据约定](docs/datasets.md)

[E162半步/全步重复各阶段](experiments/E162_repeat_half_residual/README.md)已完成固定dev500 Loss对照，并按用户要求创建后台原no-MIB dev2000进程。保持E161冻结权重、不训练；S0/S1/S2各执行两次，先0.5布局残差、再1.0。[Loss报告](experiments/E162_repeat_half_residual/loss_audit_dev500/report.md)；[实时状态](experiments/E162_repeat_half_residual/evaluation/output/state.json)。E161原dev2000已完成，混合Q1.031281153。以下为历史过程。

E161回传S2 best21的本地原no-MIB dev2000评价已正式启动，模型/后处理源码和原预算不变，结果保存到独立目录。实时进度见[状态](experiments/E161_kaggle_transfer/local_dev2000_v1/output/state.json)，[评价入口](experiments/E161_kaggle_transfer/local_dev2000_v1/README.md)。以下为历史过程。

E161四阶段训练已完成，远端70个文件已完整回传且权重哈希通过；最终为S2 best21。GPU占用已归零，后续回到本地。dev2000因迁移包漏掉评价配置未启动，本次未重训或启动本地评价。[回传记录](experiments/E161_kaggle_transfer/downloaded/20261123T084205Z/RETURN_TO_LOCAL.md)；[当前状态](docs/current_status.md)。以下为历史过程。

[E161 Kaggle任务](https://www.kaggle.com/code/andyyubao/iccad-e161-s1-resume-20261123)已于2026-11-23台北10:14成功提交私有GPU版本1，最近返回QUEUED；数据集已就绪，尚未确认新训练更新。本地训练仍暂停；不重复部署，状态见[当前记录](docs/current_status.md)。以下为历史过程。

[E161四阶段训练](experiments/E161_final_no_mib/README.md)已按用户要求改为全新scratch、由用户亲自启动。旧运行与预算证据归档至[重开记录](history/e161_before_user_scratch_20261122T160405Z/archive_manifest.json)，当前`prepared`、尚未启动；直接运行[PowerShell入口](experiments/E161_final_no_mib/launch_user_scratch.ps1)。架构、原36k/no-MIB数据、各阶段4h/200轮和最终自动dev2000保持。以下旧启动描述保留为历史过程。

最终主线已按2026-11-22用户决定回到原面积/no-MIB和原reference。[E151原dev2000评价](experiments/E151_frozen_role_initial/runs/evaluation_reference_dev2000_v1/README.md)后台进程已创建；[E161四阶段架构](experiments/E161_final_no_mib/README.md)已准备，训练关闭。最终成员为[原no-MIB dev2000](protocol/final_evaluation_no_mib_dev2000_v1.json)，不使用下述旧MIB面板。状态和边界见[当前结果](docs/current_status.md)、[执行约定](docs/protocol.md)。

[收尾架构与最终评价](docs/final_architecture_20261122.md)：L0三层加宽attention，S0/S1/S2各双层、阶段末端单次解码、独立训练冻结；MST夹挡监督与信息迁移方案已记录，[完整dev2000清单](protocol/final_evaluation_dev2000_v1.json)已登记，尚未启动新训练/评价。E160已完成best63（dev Loss0.335786179）；E157完整MIB dev500混合Q1.014650315、grouping92.2%，最新结果见[当前状态](docs/current_status.md)。以下启动描述保留为历史过程记录。

[E160 静态关系消息初稿](experiments/E160_static_relation_message/README.md) 已按用户授权完成准备并创建后台正式训练进程（2026-11-22）；按AGENTS未执行启动验证，运行状态以实验记录为准。在E159三个attention块中加入秩8的接收者相关静态关系消息，不增加分跳聚合；width128/四头、数据/MIB/Loss、36k/batch64/seed0、4h/200轮预算保持。仅L0，完成后冻结own-best并自动比较E159保存的总Loss与分项。未执行防御性测试、试训或启动验证，无后段/后处理/test；启动记录和日志见实验background目录。

[E159 加宽三层 RoleInitial](experiments/E159_role_initial128/README.md) 已完成50轮正常早停，best47，固定train500/dev500 Loss0.348669/0.347660，耗时79.9分钟。相对E158 best58的dev总Loss降低8.74%，482/500题改善；线长/紧凑分项略升。[完整比较](experiments/E159_role_initial128/comparison_e158/report.md)。同题对照10例和E159 Loss最差10例已保存，端口8829/8830。仅L0，无后处理或test。

[E158 三层 RoleInitial 初稿](experiments/E158_role_initial3/README.md) 已完成61轮正常早停，34343次更新，耗时89.6分钟，冻结best58；固定train500/dev500 Loss为0.381521/0.380959（含reference1.0和原重叠组×0.25）。[分项Loss、曲线与E155共同口径对照](experiments/E158_role_initial3/runs/review_results_v1/report.md) 已保存；随机dev20已按统一入口展示，端口8828。仅L0，无后段、后处理或test。

[E157 依次训练S1再S2](experiments/E157_sequential_s1_s2/README.md)：2026-11-20用户明确改为顺序训练并授权后台启动。冻结E156 best79；先训练S1编码器/共享双层S1两次（L6/L8 .2/.8），冻结own-best后，再新建S2编码器/共享双层S2两次（L10/L12 .2/.8）。完整公开Loss，reference及grouping瓶颈监督为0；仅S2 L12加0.2容差移动项，重叠模块减轻移动惩罚。两阶段各279958可训练参数、36k/batch64、各4h/200轮上限；两阶段冻结/恢复/梯度及完整MIB后处理短检通过后启动。最后唯一S2 best运行新dev500完整MIB评价，不读test。队列与日志以实验内记录为准。

[E156 两次共享S0](experiments/E156_two_s0_stage/README.md) 已用满原4小时预算，完成80轮及第81轮部分更新，best79 dev Loss0.189882已冻结。未启动S1/S2；父权重现供E157顺序训练使用。

[E155 新形状修正版L0](experiments/E155_fresh_mib_l0/README.md) 已完成，best51固定train500/dev500 Loss为0.360470/0.360367，父权重SHA已冻结。新36k、MIB形状修正监督；本轮未训练后段、未后处理。

[全新随机MIB 40k数据准备](datasets/random_mib40k_v1/README.md)：2026-11-19用户授权忽略历史重叠、重新随机划分与抽样；先扫描超过40k候选，MIB兼容筛选后补齐36k/2k/2k。新train/dev监督与公开输入分别处理、生成38k增强静态缓存和分层随机监测500；test仅公开规格兼容检查，标签继续封存。启动前短检通过，后台扫描/缓存已启动，不训练模型，状态以数据目录为准。

[E154 移除瓶颈项后重训 L0](experiments/E154_l0_no_bottleneck/README.md)：已完成53轮正常早停，own-Loss-best为第50轮，原固定train500 Loss0.372355/dev500 Loss0.368000，best已冻结；没有启动后段或后处理。新选[MIB结构挑战dev500](datasets/mib_dev500_challenge_v1/report.md)及独立CPU缓存已完成，按36k的N分布分配、公开约束复杂度择优，未改变E154原监测和选模结果。后续瓶颈监督仅S0/L4系数0.2，L0/L8/L12为0。

E153两阶段及完整MIB dev500已完成：最终500/500硬合法、MIB全满足，grouping453/500（90.6%），混合Q1.043412；未达到95%目标。[Loss与后处理审计](experiments/E153_mib_role_topology/runs/loss_postprocessing_audit_v1/report.md)记录了总Loss口径变化及9个外框异常案例。

[E153 MIB与增强题目编码](experiments/E153_mib_role_topology/README.md) 已按用户要求完成一个scratch方案的准备与检查：原MIB32k/train500/dev500不改划分；先处理数据，L0及所有后层使用增强条件，S0 reference为0.75并加入分组/成员重排。先训L0再训后段，无历史权重迁移。数据处理、全数据尺寸压力检查、两阶段梯度/恢复检查和完整MIB后处理短检均通过；用户已于2026-11-18授权启动，后台依次训练L0与后段，完成后自动原dev500完整MIB评价；实时状态见实验内记录。

[E152 只微调 S2 的 A/B 对照](experiments/E152_s2_local_repair/README.md) v2 两组已完成原36k、各10轮训练及原dev500。A保留reference（best10），B去掉（best8），最终均500/500硬合法，混合Q为1.008752/1.009692，软约束全满足425/419题，均无外框超过reference1.2倍的题。[统计与逐阶段展示](experiments/E152_s2_local_repair/runs/review_dev500_v1/report.md) 按本轮用户请求生成；旧版结果已作废。

[E151 固定 E149 初稿并重训后段](experiments/E151_frozen_role_initial/README.md) 使用冻结 E149 best52，原 36k 数据上训练题目编码器与 S0/S1/S2。以 E148 的 L4/L8/L12 目标选模，完成后在原 dev500 检查 S1 的重叠降幅及 L4 优势是否维持到 L8。实时状态见实验记录。

[E150 新初稿与原后段联合训练](experiments/E150_role_joint/README.md)：2026-11-18用户授权准备通过后后台启动：E149新角色初稿＋E148原12层后续结构，681141参数全部scratch seed0联合训练。L0/L4/L8辅助各1/15、L12为.8；L0和L4去直接overlap，L8/L12完整。只传布局跨双层单元，初稿也直接监督。原36k/batch64/4h，基础及E149角色缓存复用；只记录Loss，无后处理/Q/test。准备检查已通过，状态以实验内记录为准。

[E149 结构角色初稿独立训练](experiments/E149_role_initial/README.md)：2026-11-17用户授权准备通过后后台启动，只训练带结构角色编码的初稿生成器。两层width64、完整引用端口token、多跳/锚点角色和静态关系bias，共107739参数；无S0/S1/S2后段。初稿唯一输出直接监督，完整去掉overlap组、其余原Loss系数保持。原36k/batch64、4h/200epoch、原train500/dev500，基础缓存复用，仅新建公开role sidecar；只记录Loss，无后处理/Q/test。准备检查已通过，实时状态以实验内记录为准。

[E148 联合双层功能网络与形变反馈](experiments/E148_shared_functional_shape/README.md)：2026-11-17按用户修订完成准备并后台启动。三套双层单元按S0/S0/S1/S1/S2/S2共享调用，跨单元仅传布局，第一子层读取纯题目条件、第二子层不重复注入。保留E142原49维反馈、无空白统计，新增16维等面积形变响应；678941参数联合scratch seed0。仅4/8/12层监督，权重.1/.1/.8，4去直接overlap、8/12完整；原36k/batch64、整个候选4h上限及自动原dev500。准备检查已通过；实时状态以实验内记录为准。

研究 FloorSet 的矩形布局生成与后处理：网络生成全部模块的位置与形状，再由后处理满足固定规格和硬约束。评价同时看最终合法性、本地研究Q和完整逐例耗时。

| 入口 | 内容 |
|---|---|
| [当前结果](docs/current_status.md) | 模型对照、队列状态与结果原件 |
| [执行约定](docs/protocol.md) | 数据、预测边界、训练、评价与停止规则 |
| [研究摘要](docs/research_history.md) | 主要发现和证据局限 |
| [数据说明](docs/datasets.md) | 冻结集合、路径与共享缓存 |
| [实验目录](docs/experiment_catalog.md) | 现有源码和归档的实际位置 |
| [布局展示](visualization/README.md) | 唯一分组色/黑框/图例样式，可选中间层滑块 |
| [工作区结构](docs/workspace_structure.md) | 文件职责和保护对象 |

E119 global2 best55是历史架构/权重起点：原dev500最终500/500硬合法，混合Q **1.021568**。用户不将这次单seed结果作为可靠性能标杆；后续按同轮对照判断。竞赛交付身份仍是归档E101，本地无MIB研究Q不等于正式比赛成绩。

[E147 监督层后清除隐藏传递](experiments/E147_supervision_hidden_reset/README.md)：2026-11-17用户授权准备后后台启动。仅S2→S3、S6→S7、S8→S9、S10→S11重建隐藏输入，其余层保留隐藏传递，布局及其梯度始终传递。E142原49维反馈、pair bias/ψ、895123参数、batch64/36k、E139监督、scratch seed0及原4h/早停设置保持。训练后自动原固定dev500和Q日志；实时状态以实验内记录为准。

[E144/E145/E146 三组E142对照](experiments/E144_local_blank/README.md)：2026-11-16用户授权准备后后台依次scratch训练，每组自动原固定dev500。E144为截断至对应边长的脱离距离＋0.25边长新条带空白比例（894867参数）；E145仅移除ψ关系value、保留pair bias（869779）；E146层间仅传布局，每层从公开静态属性重构输入，不传前层或S0隐藏状态（895123）。均batch64、原36k、E139监督和原优化/早停/4h上限；准备检查已通过。整体实时状态在E144 `background/suite_state.json`，总日志 `background/suite.stdout.log`；Q逐组写日志，错误/STOP阻断后续。

[E143 四方向可移动距离实验](experiments/E143_directional_travel/README.md)：2026-11-16用户授权准备通过后后台scratch训练，仅改E142方向反馈：阈值0.1、连续首次越阈距离、到边界/对应边长双上限，保留0.25s响应、删除1.0s响应。45维动态输入、894867参数，其他E142设置及E139监督保持；准备检查通过，自动原固定dev500和Q日志。实时状态看实验内记录。

[E142 模块动态需求与关系value实验](experiments/E142_dual_geometry/README.md)：2026-11-16用户授权准备通过后后台scratch训练，batch64、895123参数、冻结36k；监督/Loss完整复用E139（S2/S6去overlap，S8/S10/S12完整，.05/.05/.05/.05/.8），不加入阻隔Loss。49维动态摘要经需求编码进入QKV，原18维pair bias保留，新增接收者相关pair value。启动前检查已通过，随后自动原固定dev500/原后处理/独立硬检查，Q摘要进入日志。实时状态以实验内记录为准，暂无新结果。

[E141 三组路径阻隔/逐层监督实验](experiments/E141_path_supervision/README.md)：2026-11-16用户授权准备并后台依次训练，batch64、scratch seed0、冻结36k；每组成功后自动原dev500后处理/独立硬检查。实验一S3/6/9/12完整监督加阻隔项；二为S1–S12逐层监督（仅S3/6/9/12含overlap）；三为组合。S0不监督、末层.8、辅助共.2、不沿用G/B×4。准备与实时状态以实验内 `preflight.json` / `queue_state.json` 为准；暂无新结果。

[E140结果](experiments/E140_frontend_gb4/result_review.md)：仅S2/S6 grouping/boundary各×4、后面三层不变；56轮正常早停、best53，原dev500后处理complete。混合Q1.039537（E139为1.016845），最终500/500硬合法，但全部软约束满足384→378题、分组违规83→106、贴边54→82；最坏Q4.311288。本轮未改善目标，不追加训练。

[E139实验与拓扑审计](experiments/E139_delayed_overlap/runs/e119_delayed_overlap_seed0/comparative_audit_v1/findings.md)：45轮正常早停、best42，原dev500混合Q1.016845，最终500/500硬合法。相对E134每三层对照，最终grouping违规104→83，是Q小幅收益的主要来源；但固定1170条受阻同组边的严格修复352→353、替代路径修复452→429，末层接触瓶颈与弱关系保真略差。完整换侧2→7次、仅3题，尚无可靠拓扑突破证据。原冻结配置见实验README；不追加训练或后处理。

[E138拓扑重规划](experiments/E138_topology_rewrite/outcome_20261115.md) 三组及原dev500后处理/结构审计已完成。A/B/C混合Q为1.028175/1.034249/1.045416，均500/500最终硬合法，但均差于E134每三层对照1.018624。用户于2026-11-15宣告本轮路线失败，不自动追加训练；原冻结源码与现场保留。

[E137实验记录](experiments/E137_standard_transformer256/README.md)：36k scratch seed0、学习题目编码＋单层标准Transformer（d256、4头、FFN1024、最终完整Loss、1,335,363参数）已于2026-11-14按用户要求终止。保留107完整轮、120375接受步与best/latest；best104的dev Loss为1.003467。进程已退出，后处理和地址诊断未执行，启动授权关闭、STOP/QUEUE_STOP保留；无最终Q结论。终止凭据在该实验`background/termination_verified.json`。

[E134实验记录](experiments/E134_supervision_interval/README.md)：E119每两层/每三层两组训练和原dev500后处理已完成。E120每三层触发门禁后，用户放弃后续E120研究，另一组未启动；失败现场和预算保留。

[E134逐层与拓扑形成诊断](experiments/E134_supervision_interval/runs/e119_stride3_seed0/layer_analysis_v1/findings.md)：best63同一dev500的S0–S12、监督层Loss分项、拓扑/几何轨迹、分片交叉验证探针与局部梯度。原权重未改，未启动新布局训练或后处理。

[E135实验记录](experiments/E135_initial_supervision/README.md)：E119 S0监督 A/B/C 及自动原dev500后处理已全部完成，混合Q为1.037931/1.031578/1.019951，均500/500最终硬合法；已有E134每三层对照为1.018624。见[结果对比](experiments/E135_initial_supervision/runs/results_review.md)；统一展示已支持S0–S12及逐层Loss，调用见可视化README。

[E136实验记录](experiments/E136_frontend_F_linear/README.md)：D（S0使用C的F、辅助权重使用B的线性3:4:5:6）已训练并自动完成原dev500，best46，混合Q1.021426，最终500/500硬合法。E134每三层与A/B/C/D的Reference／草稿／原后处理逐题统计位于该实验 `runs/statistics_dev500_v1/`。

浏览 E119 已存展示（10个普通例＋5个最坏例，可滑块查看真实12层）：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m visualization --e119 --port 8779
```

访问本机 `http://127.0.0.1:8779/`。12层示例及PNG导出见可视化README。Python环境在 `.conda/envs/iccad2026/`，数据通过 `protocol/paths.json` 定位。

2026-11-13已精简长期文档并统一可视化；旧版位于 `history/20261113_workspace_cleanup/`。根目录 `output/` 已按用户要求删除，不再作为长期结果位置。实验原始记录、权重、数据和有效缓存保留；本次整理不启动或恢复研究任务。
