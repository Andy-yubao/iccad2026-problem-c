# E163 · E151 的 S2 首次调用半步

原 E151 best45 冻结推理。S2 首次调用内部仍执行原两层，将其末端净残差乘 0.5 后作为第二次 S2 调用的输入。直接在中心坐标和 log 长宽比状态上计算。原 dev2000、无 MIB、独立硬检查和 E98.depth_budget.v1 后处理预算。训练更新数为零。

后台评估记录在 `evaluation/background/evaluation.log`。进度见 `evaluation/output/state.json`，完结后生成 `results.json`、`subsets.json` 和逐题基线汇总 `comparison.json`。

独立草稿延迟诊断见 `profile/draft.json` 和 `profile/backend.json`，含相同输入下 E151 与 E161 的 eager 前向、E151 几何反馈、固定端口形状的整网重放，以及保留原端口形状的后段重放和坐标核对。正式 Q 对照走原 eager 前向。

同机 12 例测速中，E151 和 E161 的 eager 前向均时为 287.4 和 33.6 ms，E151 单独重算几何特征为 215.5 ms。整网固定端口填充重放为 25.5 ms，草稿仅 1/12 例逐位一致。改为保留原端口形状、仅捕获后段后，E151 前向从同轮的 281.3 ms 降到 28.1 ms，12/12 例的全部 13 个布局状态逐位一致。后段重放首次捕获和预热约 1.44 s。以上为前向抽样测速，完整延迟和重放的全量 Q 待另行评估。
