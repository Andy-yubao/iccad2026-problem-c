# ICCAD 2026 Problem C

2026-10-29长期研究进行中。已将15个历史实验归入[历史区](history/20261029_model_research_consolidation/README.md)，迁移5,746个文件并逐文件核对；只清理72个可重建编译缓存，唯一权重、数据缓存和运行证据完整保留。日常区只保留冻结基线、可复用诊断及近期研究。

当前候选[E74](experiments/E74_collision_32k_posttrain/runs/completion/inspection.md)：完整dev2000草稿交面积6.302%→5.086%，1871/2000例改善；同100例外框与精确空白均下降。flash500加权Q1.022511、尾部改善，但配对worker重抽样区间仍微跨零，不宣布稳固Q升级，测试封存。[100例reference／候选／E42无评分图](experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/draft_comparison100/index.html)。

[E76已完成复核](experiments/E76_bounded_refinement/runs/completion/inspection.md)：30轮早停/best29，flash500加权Q1.017693，相对E42的worker配对区间[-0.012021,-0.004259]，完整dev2000交面积降至2.733%。但相对E74有499/500外框扩大、均值约增加1.31%；保留其收益证据，不按Q直接升级基线。

当前[E77](experiments/E77_compact_refinement/README.md)：冻结E76全部分支，把输出限制进E74自身公开预测外框，再训练新增局部修正器尝试内部重排。框不来自reference；固定`G+2I+5E`、32k、原监测/早停与终末协议。第六个、初始候选预算内最后一组。每30分钟跟进，[当前阶段](experiments/E77_compact_refinement/state.json)，[总账](experiments/E72_topology_posttrain/study.json)。E74/E76已有终末评价不重跑，E42/E50仍冻结，测试封存。

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
| [E77](experiments/E77_compact_refinement/README.md) | 固定公开预测外框＋冻结E76＋内部局部修正 |

归档E61/E65/E68/E70等对照与恢复方法见[迁移说明](history/20261029_model_research_consolidation/README.md)。[朱老师交付材料](output/2026-10-24_朱老师交付报告/README.md)保持原位。

先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。本聊天长训练授权记录在实验方案，不改写通用协议。独立环境`.conda/envs/iccad2026`，数据由[paths.json](protocol/paths.json)解析；冻结成员保持原件，checkpoint/数据/环境不入Git，布局图统一visualization，不自动启动评分界面。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [官方题目](references/official/problem_statement.md)
