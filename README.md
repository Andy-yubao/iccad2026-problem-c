# ICCAD 2026 Problem C

2026-10-29一期六个候选已完成并保留账本；用户新授权“长任务继续”，第二阶段研究进行中，已恢复30分钟跟进。二期先限3个新候选/6小时、每组2小时，旧预算继续计入18小时总上限。已归档15个历史实验并核对5,746个文件，只清理72个可重建编译缓存，保护全部权重与张量缓存。

当前开发候选[E77](experiments/E77_compact_refinement/runs/completion/inspection.md)：完整dev2000草稿交面积6.302%→2.205%，1999/2000例改善；500例均满足E74公开预测外框限制，精确空白继续下降。flash500加权Q1.017483，相对E42的worker配对区间[-0.013842,-0.003141]。最差单例Q2.163867劣于E42的1.927279，不能声称全指标占优或正式未见成绩，不自动替换冻结E42/E50。[100例reference／E77／E42图](experiments/E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0/draft_comparison100/index.html)。

[E76已完成复核](experiments/E76_bounded_refinement/runs/completion/inspection.md)：30轮早停/best29，flash500加权Q1.017693，相对E42的worker配对区间[-0.012021,-0.004259]，完整dev2000交面积降至2.733%。但相对E74有499/500外框扩大、均值约增加1.31%；保留其收益证据，不按Q直接升级基线。

[E77](experiments/E77_compact_refinement/README.md)已完成40轮、own-Loss best38；固定E76与E74预测外框，只学新增局部修正器。原预算保留；恢复训练阶段实测35.94分钟，另保守计入首次未知中断77.16分钟。训练、完整dev、一次flash500和100图已完成；最后报告的整数序列化错误仅通过保存结果汇总修复，冻结源码与终末评价不变。[完成状态](experiments/E77_compact_refinement/state.json)，[一期总账](experiments/E72_topology_posttrain/study.json)。一期结束时的关闭记录保留，本次新授权另开下述二期；测试封存。

新授权的[E78](experiments/E78_connectivity_posttrain/README.md)从E77优秀权重继续训练既有最后修正器，源E76和E74预测框不变。保存结果否定了简单压低原始分组/线长代理的充分性；新组合提前协调相容等面积MIB组形状，并加入固定模块间通道过填压力。五例初始形状协调增加4例重叠，负面证据保留；固定Loss/CUDA/公开规格/框/真实Dropout Adam恢复预检已通过，进入正式训练。[二期账本](experiments/E78_connectivity_posttrain/study.json)。一期结束记录不追溯改写，E77终末不重跑，测试封存。

[E78完成复核](experiments/E78_connectivity_posttrain/runs/completion/inspection.md)：MIB准备突变499→5例，flash加权Q1.015401、最大1.751561，但草稿交面积2.723%较E77的2.205%退化，不能只按Q升级。下一组[E79](experiments/E79_center_refinement/README.md)冻结完整E78及其尺寸，只新增中心修正器，以更强交面积项和一侧源次序保护尝试追回几何质量。五例预检通过；二期第二候选，每组2小时，旧预算保持。

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
| [E79](experiments/E79_center_refinement/README.md) | 当前：固定E78同形尺寸＋中心修正，五例预检通过 |

归档E61/E65/E68/E70等对照与恢复方法见[迁移说明](history/20261029_model_research_consolidation/README.md)。[朱老师交付材料](output/2026-10-24_朱老师交付报告/README.md)保持原位。

先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与[AGENTS.md](AGENTS.md)。本聊天长训练授权记录在实验方案，不改写通用协议。独立环境`.conda/envs/iccad2026`，数据由[paths.json](protocol/paths.json)解析；冻结成员保持原件，checkpoint/数据/环境不入Git，布局图统一visualization，不自动启动评分界面。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [官方题目](references/official/problem_statement.md)
