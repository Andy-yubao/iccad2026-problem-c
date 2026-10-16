# 官方输出评分

本模块复用官方评测标准，不包含训练 Loss、可微代理或搜索目标。官方实现原样保存在 `third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py`。

- `official.official_evaluator()`：加载原版评测器，包含官方指标与 `compute_total_score` 总分聚合。
- `evaluation.score_official_output(case, positions, runtime_seconds, median_runtime)`：直接调用官方 `evaluate_solution`，返回原始指标，时间因子使用显式传入的实际耗时和匹配的中位时间基准。
- `evaluation.evaluate_case(case, positions, runtime_seconds)`：延续既有本地对照，调用官方 runtime=median_runtime=1，质量与实际耗时分列，并做独立硬约束及非负坐标复核。额外复核与官方原始可行性标志分列，不能将本地质量分称为正式比赛分数。
- `evaluation.summarize_variants()`：按官方加权函数汇总已有运行，不加载额外数据；失败仍计惩罚，不只统计成功样本。
- `checks.independent_hard_checks()`：对官方硬约束作独立复核，不是训练目标。

评分需要包含参考指标的评测样本；求解器只接收数据模块提供的合法输入。若将来实验需要训练 Loss，应留在该实验内；它的权重和公式不改变本模块的官方打分。

缺少正式评测的逐例中位耗时及完整流程时，只能报告相应的本地对照。规则和指标解释见 [通用赛题导读](../docs/research_guide.zh-CN.md) 与 [官方题目](../references/official/problem_statement.md)。
