# S3之后非监督层的结构审计

2026-11-15用户授权。E134每三层best63，同一dev500；不训练、不改权重、不运行后处理/Q，不读test。

- [结论](findings.md)：换位、固定初始阻隔、贴边、最大空隙与冻结反馈干预。
- [完整表](report.md)、[逐题逐层表](per_case_layers.csv)、[数值](summary.json)。
- [主方案](plan.json)、[结果后追加的敏感性方案](followup_plan.json)、[验证](verification.json)。
- `cases/`保存全部500题的事件轨迹；`*_states.npy`、`*_losses.npy`保存固定权重干预输出。所有监督层Loss均用原完整公式。
- `audit_curves.png`为总体统计；布局PNG复用唯一分组色/黑框/底部图例渲染。示例选择见`example_selection.json`，总体统计优先于个例。预定的“最大空隙仍保留80%以上”例没有符合者，未补选一个冒充。

运行源码在实验根目录：`audit_unsupervised.py`（观察/干预）、`audit_unsupervised_followup.py`（次级控制）、`audit_unsupervised_report.py`（汇总）。完整换侧为离散输出代理；反馈干预改变多个计算入口，只解释当前模型的局部依赖，不能直接外推到重训或最终Q。
