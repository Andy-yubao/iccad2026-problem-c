# ICCAD 2026 Problem C

FloorSet Challenge 赛后研究工作区。先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与 [AGENTS.md](AGENTS.md)。

2026-10-20当前研究：[E18快速后处理](experiments/E18_fast_postprocess/README.md)。用户授权自主长期运行，先固定E17 best51优化后处理，目标为同一v5开发500例加权Q<1.04、平均单例推理+后处理<0.25秒且全部硬合法；充分检验后处理仍不足后才调整模型重训。E17已完成：500/500硬合法、加权Q=1.060518、平均1.915333秒，[完成检查](experiments/E17_e10_geometry_v5/runs/20261020_e10_geometry_v5_seed0/analysis.md)。

2026-10-20：[E19 MIB一致性模型](experiments/E19_mib_consistent_geometry/README.md)已完成62轮早停/best54，500硬合法，但同版后处理加权Q1.075417、平均1.999252秒，较E17退化，唯一best冻结。[E20快算法联合评价](experiments/E20_mib_fast_pipeline/README.md)与[E21原保留模型对照](experiments/E21_original_model_fast/README.md)串行进行；[E20日志](experiments/E20_mib_fast_pipeline/runs/20261020_background/stdout.log)。尚未同时达到完整500例两项目标；测试封存，原证据保留。

2026-10-20已启动后台流水线：[E17：E10几何训练方法在v5数据上的对照](experiments/E17_e10_geometry_v5/README.md)。497,299参数、batch32、持续参考几何/接触监督，采用四档学习率1e-3/2.5e-4/1e-4/5e-5；独立缓存已完成，已确认正式训练更新；最多4小时/200轮，结束自动执行固定best开发500例验证（4线程），不持续监测。[日志入口](experiments/E17_e10_geometry_v5/runs/20261020_background/stdout.log)。E16延后衰减模型1已完成，模型2由用户停止，不自动恢复。下述E16/E15为历史进度。

2026-10-20：[E16延后参考监督衰减](experiments/E16_reference_curriculum/README.md)新增两组497,299参数对照，分别第16/21轮开始15轮过渡；用户授权后台依次训练，每组最多4小时，复用v5数据与缓存，只确认启动后结束助手任务。本轮不调整或启动后处理，测试保持封存。下述10月19日准备状态为历史记录。

2026-10-19 最新方向：[E16原参考监督与质量目标渐进对照](experiments/E16_reference_curriculum/README.md)。放弃教师集，保留原参考监督和MIB面积修正规则，在原三分片池内重抽16k/2k/2k，按实际模块数均衡。新清单已完成；用户中断缓存后，准备阶段改为4进程、每进程单线程，保留已完成扫描和1,024例缓存。三组训练合计8小时、准备及终末评价另计；未自动恢复全量任务或开始正式训练。下述E15记录保留作历史状态，不作为继续教师集训练的指令。

2026-10-19：[E15从头训练与质量目标渐进对照](experiments/E15_quality_curriculum/README.md)三组采用共享数据缓存，训练验证均不做后处理，固定500例原始布局代理选模；500例后处理仅允许训练结束后对各组冻结best执行一次。第一组由用户结束、冻结第66轮best；后两组从头训练，采用三档学习率及15/3/8调度规则。第一组CPU评价可与后两组GPU训练并行；长任务由用户执行，学生测试继续封存。

2026-10-19：[E14尾部稳定性](experiments/E14_stable_teacher/README.md)已通过MIB修正任务的开发100＋另500复核：600/600硬合法、无Q>1.8、零退化；500例平均Q 1.054594→1.039376、最大Q 2.648058→1.458786，平均求解1.274→1.653秒。原网络权重未重训，旧一万例教师标签不进入新版库。新任务按[derived_workset_v3](protocol/derived_workset_v3.json)登记train16k/dev2k/test2k目标，测试保持封存；原始任务参照仍保留E10。见[E14结论](experiments/E14_stable_teacher/analysis.md)与[生成入口](experiments/E14_stable_teacher/launch.md)。

当前基线为 [E10：E7 模型＋E8 后处理](experiments/E10_integrated_baseline/README.md)：冻结 attention_geometry 模型，配合 conic_robust 锥优化与可行种子救援。继承的两组开发100例均全部硬合法，平均中性时间Q为1.211556/1.232217；这是已有开发结果，不是正式比赛分数或独立泛化证明。E9候选未通过另一组开发复核，不纳入。

日常实验区保留 E10、[E11短预算机制诊断](experiments/E11_short_budget_diagnosis/README.md)、[E12蒸馏准备](experiments/E12_distillation_readiness/README.md)和[无训练硬约束 baseline](experiments/baseline/README.md)。E12在两组开发100例上，以教师几何/实际接触/MIB类别配轻量投影，通过零误差与2%几何误差门槛；修复加通信约0.13～0.14秒，尚无学生模型成绩。E10未改，未启动新训练。其他实验与必要证据已移至[历史归档](history/20261018_E10_consolidation/README.md)，原始运行事实不改写；不自动恢复退役训练或未完成测试。[实验索引](experiments/README.md)。

[E13蒸馏实施](experiments/E13_solver_distillation/README.md)已提供几何/结构同预算配置、教师标签质量审计、可恢复训练和真实学生评价；[启动入口](experiments/E13_solver_distillation/launch.md)。仅完成五例短更新与停止恢复检查，正式长任务由用户启动。

独立环境 `.conda/envs/iccad2026`，见 [environment.yml](environment.yml)。唯一数据位置由 [protocol/paths.json](protocol/paths.json) 解析；[split_v1.json](protocol/split_v1.json) 与历史 [workset_v2.json](protocol/workset_v2.json) 保持冻结。新派生任务后续采用 [derived_workset_v3.json](protocol/derived_workset_v3.json) 的16,000/2,000/2,000目标，测试默认封存且尚未物化。历史测试暴露及中断范围见[协议](docs/protocol.md)。

```powershell
& .\.conda\envs\iccad2026\python.exe -B scripts/verify_workspace.py
```

数据处理复用 [datasets/](docs/datasets.md)，官方评分复用 [scoring/](scoring/README.md)，布局图统一使用 [visualization/](visualization/README.md)。模型、Loss、构造、搜索、合法化独立归各实验。官方资料与论文见 [references/](references/README.md)，未修改上游在 third_party；官方代码 Apache-2.0，数据 CC BY 4.0，保留原署名。

本地 `main` 与 GitHub [Andy-yubao/iccad2026-problem-c](https://github.com/Andy-yubao/iccad2026-problem-c) 的 `origin/main` 管理源码和必要证据。数据、环境、缓存、checkpoint 与凭据不进入 Git；E10 运行依赖本机保留的冻结权重，详见实验 README。[Git 约定](docs/git.md) · [质量目标](docs/aim.md) · [题目全文](references/official/problem_statement.md)。
