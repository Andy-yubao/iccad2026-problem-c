# 最终成果入口

唯一求解器：[E101 C5＋原完整后处理](../../experiments/E101_final_c5_solver/README.md)。

[设计说明](../../experiments/E101_final_c5_solver/DESIGN.md) · [固定训练与最终评价计划](../../experiments/E101_final_c5_solver/plan.json) · [实时运行状态](../../experiments/E101_final_c5_solver/runs/20261103_final64k_seed0/state.json)。

64k最终重训已完成：L best31，F best12，唯一组合权重已冻结，目前进入完整评价，尚无新的最终测试成绩。冻结后先dev500，再评价原local_test2000，结果与逐例完整时间留在该run的evaluation/local_test中；不根据测试结果调整模型或后处理。原C5历史成绩不是本次重训成绩。初次评价的无关新增文件清单问题已核对、保留失败并修复检查边界，未更换权重或求解算法。

旧报告与旧实验已归历史目录，完整证据和权重保留。
