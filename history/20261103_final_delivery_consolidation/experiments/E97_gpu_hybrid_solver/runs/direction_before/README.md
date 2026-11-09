# ICCAD 2026 Problem C

当前方向：冻结布局器L，训练可独立调用的全局修正器F；早期只以一次F作为主要判据，多次调用作辅助诊断。

E95 直接外框 8.46 候选已确认失败，完整证据保留。E96 完全压缩、外框权重0.5的训练与评价已完成：单次草稿交面积1.0854%、平均N5=8.342，flash加权Q1.026523，未升级冻结E90；本轮在9872步上限结束。[实验入口](experiments/E96_port_compression/README.md) · [实时状态](experiments/E96_port_compression/state.json)。

## 保留的基线与当前实验

| 目录 | 保留用途 | 已保存同dev500结果 |
|---|---|---|
| [E42](experiments/E42_joint_regularization/README.md) | 冻结draft＋flash | Q 1.025860 |
| [E50](experiments/E50_anytime_quality/README.md) | 质量优先pro，约5秒预算 | Q 1.012043 |
| [E74](experiments/E74_collision_32k_posttrain/README.md) | 较强布局草稿和尾部参照 | flash Q 1.022511 |
| [E90](experiments/E90_auxiliary_weight_placer/README.md) | 当前修正器的冻结L来源 | flash Q 1.022348 |
| [E81](experiments/E81_multisource_refinement/README.md) | 可独立调用的旧神经修正器 | flash Q 1.016242 |
| [E85](experiments/E85_cooperative_space_refinement/README.md) | 神经修正＋确定性几何算子对照 | cooperative flash Q 1.014672 |
| [E95](experiments/E95_single_pass_corrector/README.md) | 已失败的直接外框8.46候选 | flash Q 1.251831；早期归一化版本 Q 1.031633 |
| [E96](experiments/E96_port_compression/README.md) | 当前独立F：完全压缩，外框权重0.5 | 已完成；flash Q 1.026523，N5目标未达 |

这些是自适应开发结果、不同预算与流程的参照，不是统一排名、正式比赛分数或独立泛化结论。冻结E42/E50不因较低点估计自动升级。

## 工作入口

- [执行协议](docs/protocol.md) · [工作区结构](docs/workspace_structure.md) · [GPU训练工程](docs/gpu_training.md)
- [数据版本与位置](docs/datasets.md) · [Git约定](docs/git.md) · [实验索引](experiments/README.md)
- [官方题目](references/official/problem_statement.md) · [研究导读](references/research_guide.zh-CN.md)
- [本次整理与迁移清单](history/20261101_research_consolidation/README.md)
- [2026-10-30项目全貌报告](output/2026-10-30_项目研究全貌报告/README.md)（历史快照，不代表当前执行状态）

独立环境`.conda/envs/iccad2026`；数据路径由protocol/paths.json解析。测试封存，无评分界面或自动唤醒。实验执行状态与精确数字以所属runs记录为准。
