# 当前基线与近期研究

**2026-11-01：[E95已按用户要求终止](E95_single_pass_corrector/runs/20261101_single_call_seed0/stop_20261101.json)。** 因用户认为Loss异常，安全停在5轮／2586步、23.8分钟，best1；完整恢复状态保留、进程退出、GPU释放。未评价或作图，STOP保留，不自动续跑，异常原因待诊断，测试封存。[状态](E95_single_pass_corrector/state.json)。

**2026-11-01：[E94已完成](E94_posttrain_lr_reset/runs/20261101_lr1e3_seed0/qualification/report.md)。** 后训练1e-3起，无保护，33轮/65.9分钟正常早停、best30 Loss0.323142；500例加权Q1.025529、最大2.281252、500/500合法，未证实优于E90。前三轮过冲，较E93 C Loss改善约2%但Q差区间跨零；4进程×2线程及同100图完成，无追加训练，测试封存。

**2026-11-01：[E93两支验证完成](E93_matched_posttrain/runs/evaluation/report.md)。** C12轮/best10、G19轮/best16；加权Q1.026230／1.024601，均500/500合法，未证实优于E90。G改善覆盖与交面积，外框和最差Q有取舍；完整dev2000、各一次flash500及各100同例图完成，4进程×2线程核验，无自动续训，测试封存。

**2026-11-01：[E92已完成并按门槛失败终止](E92_two_stage_placer/runs/completion/report.md)。** 2h／55轮／55262步、F best15，Loss0.328326持平E90，但同例加权Q1.031269、最差Q2.154966及覆盖尾部退化；500/500后处理合法，完整dev2000和同100草稿图完成。未进入修正研究，每小时跟进已删除，STOP及全部证据保留，测试封存。

**2026-10-31：[E91异常停止](E91_e74_32k_scratch/runs/failure_20261031/README.md)：41轮／41372步、best41 Loss0.353941；梯度保护在Adam前拦截坏更新，best41另有严重失稳样例。加速前后同RNG复现一致，余约37.6分钟保留，不自动重试或Q，测试封存。**

**2026-10-31：[E89两分支已完成](E89_spatial_neural_refinement/runs/completion/inspection.md)：spatial19轮/best18，pooled13轮/best13，Q1.024304／1.025208，均未胜E90；稀疏组无改善，空间机制收益未证实。不升级基线，无活跃训练，测试封存。**

**2026-10-31：[E88](E88_sparse_training/README.md)已完成11轮／5500步、23.6分钟早停；best0保持E90起点，完整dev2000已完成，Q复用E90，未产生更好的已选模型。新增2268成员全部见过，测试封存，不自动启动E89。**

**2026-10-31：[E90](E90_auxiliary_weight_placer/README.md)已完成14轮／7000步及验证，best12、Q1.022348，500/500合法；严重覆盖尾部改善，整体/稀疏组仍逊E74，Q差区间跨零。[完成报告](E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/inspection.md)，测试封存，不自动开始E88/E89。**

**2026-10-31：[E87](E87_simple_supervised_placer/README.md)已按用户要求安全停训，第14轮／7000步、best11，完整断点与累计预算保留，已完成总2线程的best11 dev500 Q评价：加权1.024171，500/500合法，与E74/E42配对区间跨零；训练保持暂停。** [E88](E88_sparse_training/README.md)已完成48k扫描、选出2268个新增成员；E87/E88/[E89](E89_spatial_neural_refinement/README.md)后续执行等待用户恢复，测试封存，自回归排除，E86暂缓。

**2026-10-30 [E85完成核查](E85_cooperative_space_refinement/runs/completion/inspection.md)。** 协同草稿交面积1.942%对control 2.120%，弱端口稀疏3.025%对3.309%；最终Q增益不确定，主要重叠差来自固定算子直接作用，尚未解决稀疏布局结构。两组正常早停、200张四阶段图核验，无活跃训练，测试封存。

**本轮最小策略归因完成，候选数已用满，自动跟进已关闭。** [综合报告](E84_loss_control/runs/completion/strategy_attribution.html)：[E83](E83_strategy_attribution/runs/inspection.md)支持当前已学F的显著Q贡献；[E84](E84_loss_control/runs/completion/inspection.md)支持新增几何目标包的显著低重叠草稿作用，未证明显著最终Q作用；更强来源也无可靠最终Q增益。未分离的规模/数据/标签效应保留为未知，不自动增加新组。此前[四阶段图册](E81_multisource_refinement/runs/20261029_placer_corrector_audit/index.html)、[核查报告](E81_multisource_refinement/runs/20261029_placer_corrector_audit/report.md)及[五问核查](E81_multisource_refinement/runs/20261029_placer_corrector_audit/five_questions.md)保留，测试封存。

2026-10-29日常区保留以下十五个目录；其余15个实验见[归档](../history/20261029_model_research_consolidation/README.md)。旧源码/日志不因迁移改写，旧入口复现先恢复原路径，不恢复旧控制器。用户各次新授权阶段另记账，不覆盖旧候选的结题证据与费用。

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
| [E78](E78_connectivity_posttrain/runs/completion/inspection.md) | 完成：准备形状跳变显著减少、Q均值更好，草稿重叠退化 |
| [E79](E79_center_refinement/runs/completion/inspection.md) | 完成27轮/best26，尺寸保持、交面积2.358%，Q1.016383 |
| [E80](E80_modular_transfer/runs/inspection.md) | 完成：同一冻结修正包迁移小型E42，Q1.017329、153.7万参数；尾部退化，训练0更新 |
| [E81](E81_multisource_refinement/runs/completion/inspection.md) | 完成复核：171.3万参数、交面积2.491%、Q1.016242、最大1.825773；尾部仍有失败 |
| [E82](E82_global_pressure_refinement/runs/completion/inspection.md) | 完成复核：28轮/best12，交面积2.445%稍降，但Q1.018035/最大2.056040回退；保留E81 |

[一期总账](E72_topology_posttrain/study.json) · [二期当前账本](E78_connectivity_posttrain/study.json) · [E77同100图](E77_compact_refinement/runs/20261029_e76_compact_corrector_seed0/draft_comparison100/index.html)。用户已重新授权长任务，30分钟跟进恢复；固定Loss选唯一best，Q辅助，测试封存，不自动更换E42/E50或开启评分流程。
