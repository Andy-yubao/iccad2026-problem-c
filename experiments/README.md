# 当前基线与研究实验

2026-10-29整理后保留以下目录；历史材料移入[归档](../history/20261029_model_research_consolidation/README.md)，其中E63的后训练control、E58冻结扩展数据等仍可只读复用，不恢复旧控制器。

| 实验 | 用途 |
|---|---|
| [E42_joint_regularization](E42_joint_regularization/README.md) | 冻结draft best51＋flash基线；唯一共用权重与v5缓存 |
| [E50_anytime_quality](E50_anytime_quality/README.md) | 冻结pro基线，复用E42 |
| [E61_explosion_replay](E61_explosion_replay/README.md) | 稳定大模型候选、数值故障与实际步长保护证据 |
| [E65_geometry_messages](E65_geometry_messages/runs/review.md) | 几何消息候选和容量对照 |
| [E68_port_tokens](E68_port_tokens/runs/inspection.md) | 完整端口组合失败与输入审计 |
| [E69_case_loss_statistics](E69_case_loss_statistics/README.md) | 同例Loss、弱端口及位置错配诊断 |
| [E70_reference_task_loss](E70_reference_task_loss/runs/inspection.md) | 更强避碰与外框扩大取舍 |
| [E71_density_reference_curriculum](E71_density_reference_curriculum/runs/inspection.md) | 最近紧凑草稿候选、尾部失败反例 |
| [E72_topology_posttrain](E72_topology_posttrain/README.md) | 首轮扩大模型；5575步安全保护，保留失稳证据 |
| [E73_wide_robust_posttrain](E73_wide_robust_posttrain/README.md) | 当前组合：保留192宽及同Loss，固定已学图聚合参数、限制其他模块实际步长 |

E72每30分钟自动跟进，读取[state.json](E72_topology_posttrain/state.json)和[study.json](E72_topology_posttrain/study.json)。用户要求优先兼容组合，至少尝试扩大模型，基于train/dev Loss考虑32k或Loss对齐。训练选唯一固定Loss best，Q仅辅助；默认展示同100例reference/current/E42无评分图。基线未升级，封存测试未读。
