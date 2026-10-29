# 当前基线与近期研究

2026-10-29日常区保留以下十一个目录；其余15个实验见[归档](../history/20261029_model_research_consolidation/README.md)。旧源码/日志不因迁移改写，旧入口复现先恢复原路径，不恢复旧控制器。用户新授权的二期另记账，不覆盖一期六候选的结题证据。

| 实验 | 用途 |
|---|---|
| [E42](E42_joint_regularization/README.md) | 冻结draft best51＋flash，Q1.025860 |
| [E50](E50_anytime_quality/README.md) | 冻结pro，共用E42权重，独立预算 |
| [E69](E69_case_loss_statistics/README.md) | 逐例Loss、弱端口与稀疏子群诊断 |
| [E71](E71_density_reference_curriculum/runs/inspection.md) | 最近紧凑草稿对照 |
| [E72](E72_topology_posttrain/README.md) | 宽模型数值失败及本轮长期总账 |
| [E73](E73_wide_robust_posttrain/runs/inspection.md) | 稳定宽模型完成、Loss/Q错位与泛化不足证据 |
| [E74](E74_collision_32k_posttrain/runs/completion/inspection.md) | 草稿明确改善、Q均值优于E42的候选，配对区间微跨零 |
| [E75](E75_collision_box_finetune/README.md) | 3374步数值保护失败，保留漂移证据 |
| [E76](E76_bounded_refinement/runs/completion/inspection.md) | 完成：Q1.017693/重叠改善，但499/500外框扩大 |
| [E77](E77_compact_refinement/runs/completion/inspection.md) | 完成40轮/best38，Q1.017483，紧凑且交面积降至2.205%；尾部仍退化 |
| [E78](E78_connectivity_posttrain/README.md) | 二期预检通过：从E77后训练，MIB协调与锚点通道压力 |

[一期总账](E72_topology_posttrain/study.json) · [二期当前账本](E78_connectivity_posttrain/study.json) · [E77同100图](E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0/draft_comparison100/index.html)。用户已重新授权长任务，30分钟跟进恢复；固定Loss选唯一best，Q辅助，测试封存，不自动更换E42/E50或开启评分流程。
