# E25：当前最佳整合基线

冻结E17的497,299参数best51模型，采用E24 `steady`原生快速后处理。当前比较基准为加权Q **1.039577677**、平均求解 **0.259486889秒**、硬合法 **500/500**，最大Q **2.525082328**。Q已达目标，时间仍超0.25秒约9.49毫秒，不宣称同时达标。

继承的完整[500例结果](runs/20261020_controls_steady_monitor500/results.json)及源码快照保持原字节，不因整合重跑；其中历史命令与路径按发生时保留。计时包含公开特征、CPU推理、解码、全部后处理及救援，加载和评分除外，首次原生编译另列。2进程×2库线程。结果属于自适应开发面板、MIB派生任务的中性时间Q，不是正式比赛分数；额外确认500未使用，测试封存。

模型、特征及数值求解代码独立复制自E24；只调整资源路径并将运行入口限于steady。[来源登记](provenance.json)与[迁移清单](../../history/20261020_E25_consolidation/manifest.json)保存来源。模型[训练账本](runs/20261020_e10_geometry_v5_seed0/results.json)、best/latest和原参考张量缓存同盘移入本实验，只保存一份；源训练目录已归档。best SHA256为`3de22809e831edf11b56a85647a0bd9fe625e8b7617abb6eb1f4b27969cfe00a`。

需要用户明确发起新的固定面板评价时，可使用新的tag（不覆盖旧结果）：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E25_integrated_baseline.run --tag YYYYMMDD_frozen --configs steady --count 500 --direct
```

本目录保留唯一共享缓存`runs/prepared_v5_e10_cache_v1`，仅train/dev，无测试张量；退役E23/E27/E28的历史复用关系保持原记录。[迁移缓存登记](runs/cache_migration.json)注明原整合时公开草稿缓存仅更新资源路径对应的源码身份，原始几何不重算。[原迁移检查](runs/20261020_migration_check/results.json)记录最多五例功能检查，不作为新的性能结果。

2026-10-21按用户要求保留本最佳组合及未运行E29，其他已结束实验归入[本次归档](../../history/20261021_E29_consolidation/README.md)。本模型、数值求解代码、缓存和最佳结果均未改变；依赖“E23尚未训练”这一历史前提的旧迁移检查脚本移为[历史辅助脚本](../../history/20261021_E29_consolidation/legacy_E25_migration_check.py)，不作为当前运行入口。不启动训练、推理或重评分。
