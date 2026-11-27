# E60：扩大训练规模与归一化的诊断

2026-10-27用户授权排查数据／模型扩大效果不佳、过拟合及归一化质量影响。本轮只分析已存记录，并在沿用E57/E59的同5个训练成员上做CPU前向、梯度和自由输出局部优化诊断；不启动正式训练、不运行后处理、不读取测试。

本轮诊断已完成：小数据组过拟合明显，64k固定面板更支持拟合／优化受限；总G可改善而重叠恶化。归一化的稳定性收益明确，深层图分支相对幅度更小是潜在代价线索，但既有混杂对照不足以确定其质量影响。优先建议同96×12、同16k的严格归一化单因素对照；未启动新训练。

结果与可复现分析脚本见[runs/20261027_saved_and_probe/analysis.md](runs/20261027_saved_and_probe/analysis.md)、[analyze.py](analyze.py)、[probe.py](probe.py)和[structural_probe.py](structural_probe.py)。运行命令为项目独立环境的 `python -B -m experiments.E60_scaling_diagnosis.analyze`、`python -B -m experiments.E60_scaling_diagnosis.probe` 与 `python -B -m experiments.E60_scaling_diagnosis.structural_probe`。精确调用、来源哈希和输出在同一runs目录。

模型、批处理、Loss、诊断和缓存消费代码独立复制自E59，来源哈希保存于provenance.json，不跨实验导入算法。只引用既有缓存及原checkpoint，不修改或复制checkpoint。冻结权重开关归一化会造成分布变化，只能诊断分支敏感性，不能替代从头训练的归一化消融。自由输出优化不更新网络，只检验5例输出空间是否仍存在改进方向，不证明网络能学到该改进。
