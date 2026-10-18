# 实验索引

当前任务为[E9：面向 E8 后处理的模型改进](E9_model_postprocess_alignment/README.md)：目标＜1.20未达。中心修正候选dev100小幅改善至1.210118，但固定confirm100退化至1.239278；两组全硬合法，候选不提升主线、不追加同方向训练，保留旧模型＋E8和全部证据，测试未读。[完整冻结复核](E9_model_postprocess_alignment/runs/20261018_center_refiner_confirm100_seed0/analysis.md)。

E8锥优化＋可行种子救援完整验证200例，全部硬合法，两组质量均值/尾部改善，推理＋后处理加速8.66/10.14倍，作为E9后处理参照。E7冻结，额外组已用于修订，属于开发证据。[最终完整对照](E8_fast_robust_postprocess/runs/20261018_robust_confirmation100_seed0/research.md)。

当前[E7已终止](repair_friendly_draft/conclusion.md)，停止采用本轮新增占用Loss，取消未完成组合及后续评价；保留既有geometry目标、E6冻结模型/后处理和各实验运行证据，不自动恢复任何历史路线。[短期计划](short_term_plan.md)只保留E6已完成阶段的背景；新任务由用户另行指定。

最新[软约束图集与验证](graph_proposal/runs/20261017_soft_constraint_gallery_seed0/analysis.md)：训练/验证/已访问测试各10图；5例dev提高B/G局部代价权重后加权Q由4.011降至3.284，4胜1平，支持软权衡不足。原默认权重未改，测试图复用既有结果；[浏览30张图](graph_proposal/runs/20261017_soft_constraint_gallery_seed0/index.html)。

已按用户指令完成[阶段A原始提议独立评价](graph_proposal/runs/20261016_raw_proposal_train100_dev100_seed0/analysis.md)：train100/dev100各覆盖100worker，GPU推理，不运行后处理。模型有粗略全局结构，但dev中心误差为自身尺度65.67%，94.62%的块涉及重叠；固定形状平移必要下界中位51.07%。当前模型尚不满足“只需微调”，保留式修复需先用受控扰动校准；B尚未执行。

最新[确定性容量与后处理诊断](graph_proposal/runs/20261016_train5_capacity_diagnosis_seed0/analysis.md)：5例train上的小/大模型记忆误差接近零，最终Q仍不佳；完美参考提议经同pack后Q从1.179变2.352，证明后处理是瓶颈之一。仅为有界机制诊断，未扩大正式训练或重开test。

确定性模型 v2 重训与测试均已按用户要求停止，进入 [结果分析](graph_proposal/runs/20261016_frozen_best_test2500_seed0/analysis.md)：完整 train20,000/dev2,500 MSE=0.07487/0.07508，无明显过拟合迹象；冻结 best step7,250 的部分测试957例全部硬合法，零兜底，加权中性时间 Q=3.85918。只覆盖39/100 worker，不代表完整测试，也不与历史dev5直接比较。已访问测试须披露复用；本轮不续训、不补跑测试或dev100。

[第一轮短期计划原件](../history/20261017_direction_review/README.md)包含三项机制探索、一项组合检验和一项局部重构探索。E1/E2/E3/E5 短探针已完成；E1 保留确定性参照、暂停 flow/E4，尚未证明收敛或最终质量优势。E3 仅单 train 样本训练接口检查。首轮dev5阶段未读取local_test，后续v2的测试暴露见上。

| 实验 | 状态与证据 | 入口 |
|---|---|---|
| E9 面向 E8 后处理的模型改进 | 目标未达；中心修正dev100改善、确认开发组退化，不提升该候选，保留旧模型＋E8，测试封存 | [研究入口](E9_model_postprocess_alignment/README.md)、[完整复核](E9_model_postprocess_alignment/runs/20261018_center_refiner_confirm100_seed0/analysis.md)、[历史设计](E9_model_postprocess_alignment/design.md) |
| E8 快速稳定后处理与模型联调 | 最终同配置200例硬合法，均值/尾部改善，推理＋后处理8.66/10.14倍；更高绝对质量目标未全部达到 | [实验入口](E8_fast_robust_postprocess/README.md)、[最终完整对照](E8_fast_robust_postprocess/runs/20261018_robust_confirmation100_seed0/research.md) |
| E7 降低修复难度的草稿 | 用户终止；不采用新增occupancy项或矩形架构。前三组同10例检查完成，组合不补训，50例目标未验证；保留源码/权重/证据，禁止自动恢复 | [实验入口](repair_friendly_draft/README.md)、[阶段结论](repair_friendly_draft/conclusion.md)、[历史设计](repair_friendly_draft/design.md) |
| E6 可修复提议与意图感知修复 | 五组模型训练/评价已完成；新架构＋几何Loss冻结，全局后处理固定dev100 Q=1.262626（旧3.636240），100/100硬合法、97/100单例达标，当前目标已达成；只输出几何，测试封存 | [实验入口](intent_aware_repair/README.md)、[设计](intent_aware_repair/design.md) |
| 无训练硬约束基线 | 固定 dev5 四组 20/20 硬合法，search 中性时间质量分 1.9090；保留原版 SA 失败输出和五例诊断作为对照 | [说明](baseline/README.md)、[实现](baseline/optimizer.py)、[结果](baseline/runs/stage1_dev_run1/results.json) |
| E1 图条件布局提议 | v2训练/测试已停止；完整train/dev MSE=0.07487/0.07508，部分test957例全合法、零兜底、中性时间Q=3.85918；覆盖39worker，不与旧dev5直比。flow暂停 | [当前分析](graph_proposal/runs/20261016_frozen_best_test2500_seed0/analysis.md)、[历史短预算对照](graph_proposal/runs/20261016_short_budget_comparison_seed0/analysis.md)、[实验入口](graph_proposal/README.md) |
| E2 关系与几何 | 两种几何控制均 5/5 最终合法，但高兜底、Q 饱和；暂停当前版本训练，先修几何 | [证据与停止判断](topology_geometry/README.md) |
| E3 规则候选构造 | greedy/no_soft 均 5/5 合法，Q=4.4848/6.7269；单 train 例精确覆盖 4/28，暂缓完整模仿训练 | [证据与覆盖诊断](masked_construction/README.md) |
| E4 规则训练×推理投影 | E1 flow 未显示最终收益且弱于确定性，本轮不放大生成模型，不启动新增规则训练 | [原计划快照](../history/20261017_direction_review/short_term_plan_before.md)、[本轮取舍](graph_proposal/runs/20261016_short_budget_comparison_seed0/analysis.md) |
| E5 冲突选区修复 | 坐标探针全合法，冲突选区 Q=1.9060 优于随机而弱于同增量预算全局控制；暂停局部学习 | [探针与比较边界](conflict_local_repair/README.md) |

仅复用 datasets 数据处理与 scoring 官方评分。后续新增实验必须先明确假设、对照、面板、预算和可证伪判据，遵守 [执行协议](../docs/protocol.md) 与 [结构规范](../docs/workspace_structure.md)。

已废弃方向按用户要求从工作区与旧源码归档中删除，不作为后续默认架构、参考权重或续跑起点。原始数据、官方资料、现有基线证据与独立环境保留。
