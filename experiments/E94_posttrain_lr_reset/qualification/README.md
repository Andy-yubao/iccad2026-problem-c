# E94固定500例自动验证

只在训练正常结束后冻结唯一F own-Loss best，复现500例Loss，再执行一次同例flash500、4进程×2线程及固定100例无评分控件图。E90/E92/E93 C/G旧结果只读复用。终末不选模，测试封存，不额外推理dev2000。

独立复制E93评价算法，runtime/provenance.json保留相同E90后处理哈希；本实验修改生命周期及500-only入口，去除旧研究门槛、修正器及唤醒逻辑。启动前评估源码哈希保存于runs/preflight/results.json；训练完成时冻结candidate与评估身份并保存源码。预算另限1h，STOP与未完成成员恢复规则见实验README。
