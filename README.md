# ICCAD 2026 Problem C

2026-10-29：按用户授权整理工作区并启动新一轮自主模型研究。11个历史实验（4,528个文件）移入[本次归档](history/20261029_model_research_consolidation/README.md)，逐文件核对迁移，权重和唯一证据保留。旧实验控制器不恢复。

首轮研究[E72：扩大模型＋近邻拓扑监督后训练](experiments/E72_topology_posttrain/README.md)：从E42 best51函数保持扩宽96→192，以固定G＋近邻方向分离项进行后训练，并保留图层实际步长保护。兼容思路优先组合；至少完成一次扩大模型，检查train/dev固定Loss，泛化受限再考虑32k，Loss降而Q升则优先检查目标错位。先完成五例实现检查再启动。每30分钟自动跟进，首轮状态见[state.json](experiments/E72_topology_posttrain/state.json)，当前E73见[运行状态](experiments/E73_wide_robust_posttrain/state.json)和[研究记录](experiments/E72_topology_posttrain/study.json)。尚无升级结论，封存测试不参与本轮自适应优化。

当前继续[E73：保留图聚合映射的宽模型后训练](experiments/E73_wide_robust_posttrain/README.md)。E72第5575步触发数值保护；同批次隐藏上限31→403，图矩阵累计漂移约99%～130%。五例恢复图参数可显著缓解主要失败，直接关闭图分支损害其他例。E73保留192宽和同一Loss，固定预训练图聚合参数，约束其余模块实际Adam步长。尚无升级结论，先检验稳定拟合，不把失败解释为泛化不足。

## 冻结基线

| 基线 | 实现与证据 | 同dev500结果 |
|---|---|---|
| draft＋flash | [E42](experiments/E42_joint_regularization/README.md)，96×12 best51、原steady后处理 | 加权Q 1.025860 |
| pro | [E50](experiments/E50_anytime_quality/README.md)，共用E42 draft、质量优先5秒后处理 | 加权Q 1.012043，12例较flash回退 |

两者属于不同时间预算。Q是MIB派生任务的本地中性时间开发指标，不是正式比赛成绩；开发与已暴露历史测试不能合称独立盲测。共用E42权重SHA256 `12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6`，原运行路径和v5共享train/dev缓存保持原位。

## 保留的研究参照

| 实验 | 为什么保留 | 状态 |
|---|---|---|
| [E61](experiments/E61_explosion_replay/README.md) | 数值稳定大模型、故障复现与步长保护 | dev G0.338659、flash Q1.024106；草稿重叠未胜E42，未升级 |
| [E65](experiments/E65_geometry_messages/runs/review.md) | 几何消息及匹配容量对照 | 小幅单seed收益；已结束 |
| [E68](experiments/E68_port_tokens/runs/inspection.md) | 完整端口表示及输入审计 | G0.390001、交面积8.426%，组合未胜出 |
| [E69](experiments/E69_case_loss_statistics/README.md) | 逐案例难度、弱端口/稀疏子群与局部位置错配诊断 | 保存的诊断证据 |
| [E70](experiments/E70_reference_task_loss/runs/20261028_bbox_void_audit/report.md) | 避碰与外框取舍反例 | 交面积3.820%，但外框广泛扩大，用户不接受 |
| [E71](experiments/E71_density_reference_curriculum/runs/inspection.md) | 最近紧凑草稿候选 | 交面积4.972%；同100图较E42紧凑/交面积改善，flash Q1.038751和尾部未胜出 |

所有候选与正式基线分开，原权重及证据冻结。旧长任务已停止，不因保留目录自动恢复。历史实验及E58冻结扩展数据缓存的迁移/严格复现方法见[归档说明](history/20261029_model_research_consolidation/README.md)。[朱老师交付材料](output/2026-10-24_朱老师交付报告/README.md)保持原位。

先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)和[AGENTS.md](AGENTS.md)。独立环境 `.conda/envs/iccad2026`；数据由[paths.json](protocol/paths.json)解析，冻结划分保持原件。数据、环境、缓存、checkpoint不入Git，布局图统一使用visualization，不自动打开评分界面。

[实验索引](experiments/README.md) · [Git约定](docs/git.md) · [研究目标](docs/aim.md) · [官方题目](references/official/problem_statement.md)
