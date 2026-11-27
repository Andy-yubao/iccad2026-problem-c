# 日常实验区

| 实验 | 当前用途 |
|---|---|
| [E100](E100_relation_guided_placer/runs/20261103_relation_guided_seed0/evaluation_dev500/report.md) | best61固定dev500：完整后处理全合法、零回退，Q1.031557；草稿8.09%成对交面积，未改善E98；近邻关系83.78%与坐标启发式83.54%接近，测试封存 |
| [E99](E99_attraction_repulsion/README.md) | 统一tuning.json＋一键固定100例对照/出图/缓存/断点；本组20轮Q1.013471未优于原流程1.013053，旧v3 Q1.011652保留 |
| [E98](E98_loss_weight_learning/README.md) | 96×12基底dev500 Q1.028555、全硬合法；首轮训练集反馈调权后台运行，含匹配原系数对照，dev仅冻结后终末评价 |
| [E42](E42_joint_regularization/README.md) | 冻结draft＋flash |
| [E50](E50_anytime_quality/README.md) | 冻结pro质量基线 |
| [E74](E74_collision_32k_posttrain/README.md) | 较强布局草稿／尾部参照 |
| [E90](E90_auxiliary_weight_placer/README.md) | E95冻结布局器来源 |
| [E81](E81_multisource_refinement/README.md) | 旧独立神经修正器 |
| [E85](E85_cooperative_space_refinement/README.md) | 含确定性算子的修正对照 |
| [E95](E95_single_pass_corrector/README.md) | 直接外框8.46候选已失败；历史证据保留 |
| [E96](E96_port_compression/README.md) | 完全压缩F，已完成且未升级 |
| [E97](E97_gpu_hybrid_solver/README.md) | 原8候选及第二窗口7候选已收尾；完整后处理C5 Q1.012686，全硬合法，质量/延迟尾部未达标 |

历史实验E69、E71–73、E75–80、E82–84、E86–89、E91–94已归[2026-11-01归档](../history/20261101_research_consolidation/README.md)。旧权重、运行事实与源码保留，路径由manifest解释；不因归档恢复旧计划或重复评价。

模型、Loss和算法按实验独立维护；共享数据处理、官方评分和布局绘图。规则见[协议](../docs/protocol.md)与[结构](../docs/workspace_structure.md)。

前一轮三步流程：[E97训练研究报告](E97_gpu_hybrid_solver/runs/final_report.md)。8个正式候选及唯一best评价全部完成，同3步预算最低Q1.169365，未到1.01；最后C8 Q1.233626、500/500硬合法、完整均时.14017秒/P95 .21869秒。因候选上限收尾，自动唤醒关闭，测试封存。后续完整后处理补测见下文。

此前[E96完全压缩修正器](E96_port_compression/README.md)，外框权重0.5；训练/评价完成，交面积1.0854%、N5=8.342、flash Q1.026523，未升级；regional32停用，无续训。

用户新授权的[E97旧完整后处理单线程补测](E97_gpu_hybrid_solver/runs/fullrepair_singlethread_findings.md)已完成：相同配置E85 Q1.015922，C3 Q1.013332、C6 Q1.013265；10组各500例全部合法且零回退。原弱三步流程的排名不代表完整流程排名；未达到最终质量/速度双目标，无新训练或自动唤醒。
