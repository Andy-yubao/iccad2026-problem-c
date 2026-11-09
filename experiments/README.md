# 当前实验区

- [E116](E116_first_draft_comparison/RUNNING.md)：原结构前4层 vs 新版首稿生成阶段（4个Attention＋SwiGLU块、末端全局条件读出）；冻结16k、唯一末端Loss。已获后台启动授权，original4→new_first串行，各4h/200轮；错误/用户停止不推进，无Q/后处理/test。

这里保留E98、E101–E113的现行路径与必要资产依赖，便于追溯和后续经授权复用；不代表这些实验正在运行。

- [E112](E112_deep_geometry_budget/README.md)、[E113](E113_deep_head_budget/README.md)：近A总参数预算的8图块几何/预测头加深，已授权串行后台训练并自动验证/五例展示；真实阶段见E112队列。
- [E109](E109_deep_supervision_shared/README.md)：已完成的共享头逐层监督A，用户选择的重要研究基线；[E110](E110_deep_supervision_heads/README.md)、[E111](E111_progressive_depth/README.md)也已完成训练/固定dev500。

- [E108](E108_random_reference40k/README.md)：新随机36k/2k/2k原始reference数据及图片准备，后续忽略MIB，本次无训练。

- [E101](E101_final_c5_solver/README.md)：原C5交付与不同重训权重身份。
- [E102](E102_reference_image_generation/README.md)：独立图片兴趣实验，2026-11-06已授权单次后台训练与visualization下固定100例图片对比；实时阶段见run状态。
- [E98](E98_loss_weight_learning/README.md)：已放弃调权搜索，固定基线保护。
- [E103](E103_offline_loss_alignment/README.md)：离线/全局统计与两个已完成续训。
- [E104](E104_scaling32k/README.md)、[E105](E105_gated_prenorm_placer/README.md)：完成的扩容/结构候选。
- [E106](E106_postprocessor_aware_loss/README.md)：代理完成但方向未合格，未训练布局器。
- [E107](E107_geometry_posttrain/README.md)：门禁停止，未做终末后处理。

[带日期状态](../docs/current_status.md) · [全部历史实际目录](../docs/experiment_catalog.md) · [有价值的研究结论](../docs/research_history.md)。所有旧命令只追溯，不自动恢复STOP或完成任务。
