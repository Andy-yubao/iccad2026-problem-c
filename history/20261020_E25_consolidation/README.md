# E25整合与工作区清理

清理前源码与必要证据提交`c70961d5820feddad3a81238def484a67848eca9`已推送并核验origin/main。按用户要求，日常实验区只保留[E25当前最佳](../../experiments/E25_integrated_baseline/README.md)、[E23待训练容量对照](../../experiments/E23_capacity_geometry/README.md)及baseline。

旧实验同盘移入本目录，历史命令、配置、结果、源码快照和日志保持原样。E17选定训练账本及best/latest、唯一原参考张量缓存移入E25；E24 steady胜出运行和公开草稿缓存移入E25，不重复保存。旧链接和路径须结合[manifest.json](manifest.json)定位。原实验源码保留，可用于恢复与审计，不自动续做退役方向。

精确逐文件旧新路径、字节数、时间与必要哈希见清单；数据缓存按同盘移动前后大小/mtime核验，模型权重及源码/运行证据核验SHA256。冻结protocol全部保持原字节。数据、环境与本机唯一权重不入Git。

本轮不启动训练、不读取封存测试、不重跑500例。E23训练命令只交给用户执行。

- [E10_integrated_baseline](experiments/E10_integrated_baseline/README.md)
- [E11_short_budget_diagnosis](experiments/E11_short_budget_diagnosis/README.md)
- [E12_distillation_readiness](experiments/E12_distillation_readiness/README.md)
- [E13_solver_distillation](experiments/E13_solver_distillation/README.md)
- [E14_stable_teacher](experiments/E14_stable_teacher/README.md)
- [E15_quality_curriculum](experiments/E15_quality_curriculum/README.md)
- [E16_reference_curriculum](experiments/E16_reference_curriculum/README.md)
- [E17_e10_geometry_v5](experiments/E17_e10_geometry_v5/README.md)
- [E18_fast_postprocess](experiments/E18_fast_postprocess/README.md)
- [E19_mib_consistent_geometry](experiments/E19_mib_consistent_geometry/README.md)
- [E20_mib_fast_pipeline](experiments/E20_mib_fast_pipeline/README.md)
- [E21_original_model_fast](experiments/E21_original_model_fast/README.md)
- [E22_native_geometry](experiments/E22_native_geometry/README.md)
- [E24_reference_fast](experiments/E24_reference_fast/README.md)

退役E15/E16的可重建张量缓存及无用Python缓存已清除，准备清单保留；恢复旧实验需重建缓存。[清理清单](cleanup.json)记录范围，不删除布局库、原始数据或checkpoint。当前E25共享训练缓存完整保留。
