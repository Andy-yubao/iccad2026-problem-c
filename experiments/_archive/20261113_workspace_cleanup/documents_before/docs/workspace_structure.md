# 工作区结构

2026-11-11整理：用户要求 `experiments/` 仅保留 E119 与 E120。核查全部队列complete、无存活控制器、前30分钟零写入、无junction后，E98、E101–E118、E121–E130共29个目录（145,392个文件，约256.6GB）同盘rename归档至 `history/20261111_experiments_consolidation/experiments/`；移动前后逐文件清单（路径/大小/mtime，≤8MiB含SHA256）核对一致。`datasets/static_tensor_cache_v1`、protocol/ 索引与 E119/E120 原件不动；E101隔离运行根在归档后部分深层路径超过MAX_PATH，读取须用长路径工具（见该归档README）。历史链接不批改，映射见归档manifest。

2026-11-11三处监督：`experiments/E129_global2_s0ref`与`E130_node12_s0ref`独立本地算法副本，原E119/E120架构，新增暴露已有S0用于纯reference监督，L4原Loss/L12 reference×.8，总权重.1/.1/.8。准备/短检/启动登记`output/supervision_s0_20261111`；E129主串行队列，各自launch_logs，原CPU缓存共享。（2026-11-11整理时已随收敛归档至 history/20261111_experiments_consolidation。）

2026-11-10监督首轮：`experiments/E125_global2_ref04`、`E126_global2_refsplit`、`E127_node12_ref04`、`E128_node12_refsplit`为四个独立本地算法副本；仅改变监督层reference系数，原S0只读复用。E125主控制器串行训练/原dev500；准备/缓存/短检/启动登记见`output/supervision_ref_20261110`，数据仍共享`datasets/static_tensor_cache_v1`。不自动追加其他方向。

2026-11-09 scaling16：独立入口`experiments/E122_global2_scaling16`、`E123_node12_scaling16`、`E124_residual_scaling16`，对应E119/E120/E121。独立本地算法副本，原源码/权重不改；各`plan.json`/`preflight.json`/`launch_logs`，E122主队列控制三组训练与原dev500。准备、缓存核对及启动脚本在`output/scaling16_20261109`；数据继续共享`datasets/static_tensor_cache_v1`，不重复构建。

2026-11-09最新授权：用户最终确认E121_residual_update研究布局状态S残差，原E119头仅生成S0，12几何块各自独立Linear64→3输出delta，S_k=S_(k-1)+delta、eta固定1；保留E119初始2全局图块/18维模块对MLP/状态反馈，非H残差、非共享delta头、非Node12。714615参数，新增2340。原随机delta初始化实际触发有限Loss/梯度门禁、零Adam；仅新层权重和bias初始缩放.01（非零），eta/公式/旧主干/LR门禁不变，失败原件归preparation_diagnostics。相同五训练成员缓存短检、精确递推和一次受保护更新通过，254个E119初始张量/CPU RNG对齐。仅单组scratchseed0/36k/4+12/FP32batch32/原增强LR门禁/4h或200轮；准备后独立隐藏后台启动，成功后唯一Loss-best冻结并自动原固定dev500/E98.depth_budget.v1。完整复用静态CPU缓存，不重训基线、不读test、不新增图，失败/用户停止不推进不自动重试，旧STOP保持；状态E121/queue_state.json，日志launch_logs。E119研究基线/E101交付不变。

2026-11-09最新授权：用户确认Node12具体公式并要求准备后后台启动，登记E120_node12_collision，单组seed0/36k/.2L4+.8L12/FP32 batch32/原增强LR门禁/4h或200轮；保留E119初始2层全局图交互、共享头和12层主干，将逐对18维MLP替换为逐层Node12、独立12→64 Q/K投影，V不变。tau=.25、精确四向脱离压力、log1p/计数归一化、连续项正常反传、每头lambda初值.1可学习，无B/C或碰撞图；实际725619参数。原五训练成员数学/轴交换/梯度/CUDA与一次可丢弃更新检查通过，复用完整36k/dev500静态缓存；正式scratch初始化。独立隐藏后台任务运行，成功后唯一Loss-best冻结，自动原固定dev500/E98.depth_budget.v1（非三倍预算）无MIB本地Q/硬检查/完整耗时；失败/停止不推进、不自动重试、不读test、不新增图。状态E120/queue_state.json，日志launch_logs，公式formula_spec.md；E119研究基线/E101交付及其他旧STOP保持。此前“精确公式待确认/不实施”边界已由本次明确授权覆盖。

2026-11-09最新授权：E118原E114同架构1+12监督（612307参数）→其固定dev500→E119初始生成器前增加2层无草稿几何的全局Attention＋归一化连接/分组聚合（712275参数、仍原4+12监督）→其固定dev500，串行后台各seed0/36k/4h或200轮；已完成E114 4+12 best48与原评价只读复用，不重训、不用Node26。两组同原五训练例缓存短检和一次可丢弃受保护更新通过；主干初始226张量与CPU RNG对齐，FP32/batch32/增强/LR/门禁/无MIB/全模块预测保持。完整复用现有36k/dev500静态缓存；各唯一Loss-best冻结后用原E98.depth_budget.v1（非三倍预算），不按Q回选、不读test、不新增图。独立隐藏Windows任务启动，无需等到正式更新；错误/用户停止不推进、不自动重试。控制器E118/run_queue.py，实时E118/queue_state.json，日志各自launch_logs。E114研究/E101交付不变，其他旧STOP不动。

2026-11-09：用户明确Node26采用36k和4+12监督，五训练例检查（含一次可丢弃受保护更新）通过后后台启动E117。仅本组seed0、12层/64维/4头、647155参数，替换逐对MLP为每层26维节点几何Q/K投影；FP32/batch32/增强/原Loss与门禁、4h/200轮保持。训练成功后自动冻结唯一own-Loss-best，原固定dev500/E98.depth_budget.v1无MIB本地Q及完整硬检查/耗时，再存同五训练例逐层坐标；失败/用户停止不推进、不重试、不读test。状态和日志见 `experiments/E117_node26/queue_state.json`、`launch_logs/`。E114主基线/E101交付及其他旧STOP不改。

更新：2026-11-05。当前工作区按“规则 → 接手判断 → 实验原证据”阅读；不把历史归档当待运行队列。[全部实验实际位置](experiment_catalog.md) · [当前状态](current_status.md)。

| 位置 | 职责 |
|---|---|
| README.md / AGENTS.md | 总入口 / 简短现行边界 |
| docs/START_HERE.md | 接手模型的最小背景与判断入口 |
| docs/research_history.md | 按问题组织的研究结论、失败机制、限制和证据 |
| docs/current_status.md | 带日期的运行状态快照，不取代run内机器状态 |
| docs/experiment_catalog.md | 按实际目录生成的实验索引，不推断完成状态 |
| docs/protocol.md / datasets.md / gpu_training.md / git.md | 可复用协议、数据、工程、源码规则 |
| experiments/En_xxx/ | 仅E119、E120源码、计划、来源和runs；其余实验归档于 history/20261111_experiments_consolidation |
| experiments/En_xxx/runs/ | 冻结身份、源码快照、状态、日志、best/latest、坐标、报告 |
| datasets/ / scoring/ / visualization/ | 共享数据读取、官方评分与独立硬检查、纯展示 |
| protocol/ | 原始split、冻结成员、面板及paths.json；旧索引不改；random_reference40k_v1父集合与random_reference16k_v1训练视图、研究规则 |
| references/ / third_party/ | 官方材料、阅读资料及固定上游代码 |
| history/ | 退役实验和旧文档原件、迁移清单；[归档导航](../history/README.md) |
| output/ | 面向人的交付与研究提案；[入口](../output/README.md)，提案不等于已实施 |
| .conda/ / .cache/ | 独立环境、有效缓存；目录名含cache不代表可删 |
| tmp/ | 临时工作位置；一次性准备/诊断脚本已归本次history，不作运行入口 |

## 受保护的路径和依赖

- E101（2026-11-11起位于 `history/20261111_experiments_consolidation/experiments/E101_final_c5_solver`）`assets/c5_baseline.pt` 是原C5；`delivery/model.pt`是首次64k暖启动导出；历史链路导出在 `runs/20261103_replay/sf/experiments/E101_final_c5_solver/delivery/model.pt`。不能按同名model.pt替换。
- E101 replay和chain64k的`s42/s74/sf`为隔离运行根，含共享数据junction/缓存硬链接；2026-11-05曾约定不移动，2026-11-11按用户整理指令随E101整体归档（同卷rename，硬链接保持），归档后深层路径超过MAX_PATH须用长路径工具读取。
- E42原有效缓存位于 `history/20261103_final_delivery_consolidation/experiments/E42_joint_regularization`；E58扩展缓存位于 `history/20261029_model_research_consolidation/experiments/E58_training_data_v6`。生产者归档不代表缓存失效，消费者按已登记映射读取。
- E103（归档于20261111）assets/sources/evidence是五个来源模型的保护副本；E106（同归档）独立保存三个来源推理模型与25000标签。是否“重复”不能只按文件名或内容判断，因为它们承载冻结身份。
- 每个实验的best/latest、唯一导出、失败现场、STOP、有效缓存、原始数据和环境均保护；旧STOP不清除。

## 历史位置的解释

20261103归档收纳E42/E50/E74/E81/E85/E90/E95/E96/E97/E99/E100等，E98后来整体迁回experiments，映射见该归档的`E98_restore_20261104.json`。E55–E70等主要位于20261029归档，E71–E94部分位于20261101归档；20261111归档收纳E98、E101–E118、E121–E130（experiments收敛为E119/E120）。准确路径以实验目录索引为准。

历史README、JSON和命令保留发生时的路径与授权。旧链接可能依赖原工作区，先查对应manifest；不为“修链接”批改冻结源码或运行证据。严格复现须登记新入口/读取适配或明确恢复原目录，不能绕过身份检查。

## 本次整理

原16份文档（含后补结构提案快照）和Git状态已保存到 [20261105_workspace_handoff](../history/20261105_workspace_handoff/README.md)。tmp中的一次性准备/诊断脚本按哈希迁入该归档；现行实验源码、所有runs和数值资产原位保留。本次目标是减少过期导航和临时脚本干扰，不以释放大体积磁盘为目的。

后续清理顺序：进程 → 引用/路径 → 原件快照 → 同盘迁移 → 清单与哈希核对 → 当前导航。移动/递归删除前核实绝对目标在指定工作区内。Git初始大量删除/未跟踪来自既有整理，不能据此reset、批量暂存或擅自提交。
