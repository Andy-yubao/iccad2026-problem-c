# E161 S2 best21本地原dev2000评价

用户于2026-11-23授权正式运行。权重为回传且已冻结的S2 best21，原reference/no-MIB dev2000和E98.depth_budget.v1原预算。入口仅重定向原评价器的工作目录，模型/后处理源码不变；本地原训练暂停标记不动。

启动：`python -u -B -m experiments.E161_kaggle_transfer.run_local_dev2000 --start`。状态见`output/state.json`，日志见`background/evaluation.log`，完成后为`output/results.json`和`output/subsets.json`。不训练、不解封test、不按Q选模。
