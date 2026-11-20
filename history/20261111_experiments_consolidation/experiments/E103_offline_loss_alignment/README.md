# E103：Loss对齐的离线与真实训练研究

更新：2026-11-05。已授权数据库、离线分析、方向检查、两个续训候选及固定dev500评价均完成。E98搜索仍放弃/STOP；没有待自动启动的新训练或测试。旧同题排序目标后来被用户纠正为跨题目全局统计，原数据/结果保留为历史。

## 按问题读结果

| 工作 | 设计与结果 | 原始证据 |
|---|---|---|
| 同1000题×5模型数据库 | E42/E68/E70/E71/E98增强，各一成熟best；600拟合/200调参/200检查 | [报告](runs/20261104_bank1000x5_seed0/report.md) · [来源清单](model_manifest.json) |
| 排序拟合与40题方向 | 检查排序55.61%→57.98%；小步改善、大步变坏，双步长门槛未通过 | [同一报告](runs/20261104_bank1000x5_seed0/report.md) |
| 第一轮系数真实续训 | 用户覆盖旧门槛直接授权；E98增强best71→fresh AdamW、LR1e-3、32轮早停best29 | [训练](runs/20261104_learned_v1_lr1e3_seed0/report.md) · [dev500](runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500/report.md) |
| 两来源一次refit | 仅增强best71/当前best29，同旧1000题，fit600/lambda1；后被全局问题替代，系数未训练 | [历史提案](runs/20261104_pair_enhanced_current_v1/report.md) |
| 跨题全局Loss–Q | 新2000不同训练题、排除旧1000，每题一稿，两模型各1000；1400拟合/600检查 | [报告与图](runs/20261104_global_independent2000_seed0/report.md) |
| reference×.75 | 从原增强best71，position=.75/shape=.1875，其余8项原值、不归一化；11轮早停best8 | [训练](runs/20261104_reference075_seed0/report.md) · [dev500](runs/20261104_reference075_seed0/evaluation_depth_dev500/report.md) |

## 实际训练收益的边界

| 固定dev500 / E98.depth_budget.v1 | 加权Q | 平均Q | 混合Q |
|---|---:|---:|---:|
| 增强父best71 | 1.043560018 | 1.050714113 | 1.047137066 |
| 新Loss best29 | 1.040796990 | 1.052651602 | 1.046724296 |
| reference×.75 best8 | 1.039699828 | 1.053338179 | 1.046519004 |

三者最终500/500硬合法、草稿0/500；候选先按各自固定dev Loss冻结，再评价Q，父结果通过五题精确桥后复用。混合Q仅小幅改善，平均Q变差；没有等训练量原Loss续训控制，不能把变化单因果归于系数。完整时间/几何/100图见各报告。测试未读取，不按Q重选。

全局检查Pearson：原Loss .2530、当前v1 .2576、新拟合 .2585；Spearman .3250/.3392/.4563。跨题关联不证明同题局部梯度可用；零系数可能来自共线性，不能推断物理项无意义。全局拟合系数尚未用于训练。

## 数据、资产与复用

所有统计/方向题来自train，内部检查只对拟合留出，布局模型已见过这些成员；不是未见验证或封存test。固定原split/v5成员、公开MIB平均面积变换。CPU FP32单线程模型推理、四进程标签任务、冻结E98.depth_budget.v1；不同于历史GPU路径，不能冒称逐位复现旧草稿。

assets保护五份best，sources保护匹配推理源码，evidence保护原results/config，model_manifest记录来源/哈希；不因“重复”删除。完整端口历史来源按原输入忠实复现，不改变现行压缩输入规则。后处理标签可用已存结果诊断，不自动再生成。

[当前方法思考（提案）](research_notes/20261105_fixed_postprocessor_training_objective.zh-CN.md) · [旧离线计划](plan.json) · [续训登记](finetune_learned_v1/README.md) · [全局设计](global_loss_q/README.md) · [reference .75设计](reference075_posttrain/README.md)。旧计划里的下一阶段/命令记录历史语境，不能重复启动已结束队列。
