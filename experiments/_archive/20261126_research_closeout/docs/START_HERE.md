# 接手入口

这份接手说明记录的是归档前的研究状态。现行入口见[项目首页](../../../../README.md)、[E169 求解器](../../../../solver/README.md)和[实验目录](../../README.md)。

先读根目录 [README](../README.md)，再按任务阅读 [执行约定](protocol.md)、[当前结果](current_status.md) 与 [实验目录](experiment_catalog.md)。研究判断补读 [研究摘要](research_history.md)；数据与性能任务分别读 [数据说明](datasets.md) 和 [GPU工程](gpu_training.md)。

当前主要研究模型为 E119 global2 best55，竞赛交付模型仍为 E101 原 C5。二者的数据/后处理口径不同，不混称同一模型。E133 本轮 FM 质量评价独立于原自动评价失败记录，见当前结果。

`AGENTS.md` 只作目标与导航，实验细节放实验内；状态以 `queue_state.json`、run 的 `results.json` 和实际进程为准。旧授权、STOP、预算和冻结源码不因文档整理而改变。清理后所有布局展示统一调用 [visualization](../visualization/README.md)，不再建专用界面。
