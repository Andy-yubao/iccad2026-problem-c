# ICCAD 2026 Problem C

FloorSet Challenge 赛后研究工作区。先读[执行协议](docs/protocol.md)、[结构规范](docs/workspace_structure.md)与 [AGENTS.md](AGENTS.md)。

2026-10-19：[E14尾部稳定性](experiments/E14_stable_teacher/README.md)已通过MIB修正任务的开发100＋另500复核：600/600硬合法、无Q>1.8、零退化；500例平均Q 1.054594→1.039376、最大Q 2.648058→1.458786，平均求解1.274→1.653秒。原网络权重未重训，旧一万例教师标签不进入新版库。新任务按[derived_workset_v3](protocol/derived_workset_v3.json)登记train16k/dev2k/test2k目标，测试保持封存；原始任务参照仍保留E10。见[E14结论](experiments/E14_stable_teacher/analysis.md)与[生成入口](experiments/E14_stable_teacher/launch.md)。

当前基线为 [E10：E7 模型＋E8 后处理](experiments/E10_integrated_baseline/README.md)：冻结 attention_geometry 模型，配合 conic_robust 锥优化与可行种子救援。继承的两组开发100例均全部硬合法，平均中性时间Q为1.211556/1.232217；这是已有开发结果，不是正式比赛分数或独立泛化证明。E9候选未通过另一组开发复核，不纳入。

日常实验区保留 E10 和[无训练硬约束 baseline](experiments/baseline/README.md)。其他实验与必要证据已移至[历史归档](history/20261018_E10_consolidation/README.md)，原始运行事实不改写；不自动恢复退役训练或未完成测试。[实验索引](experiments/README.md)。

独立环境 `.conda/envs/iccad2026`，见 [environment.yml](environment.yml)。唯一数据位置由 [protocol/paths.json](protocol/paths.json) 解析；[split_v1.json](protocol/split_v1.json) 与历史 [workset_v2.json](protocol/workset_v2.json) 保持冻结。新派生任务后续采用 [derived_workset_v3.json](protocol/derived_workset_v3.json) 的16,000/2,000/2,000目标，测试默认封存且尚未物化。历史测试暴露及中断范围见[协议](docs/protocol.md)。

```powershell
& .\.conda\envs\iccad2026\python.exe -B scripts/verify_workspace.py
```

数据处理复用 [datasets/](docs/datasets.md)，官方评分复用 [scoring/](scoring/README.md)，布局图统一使用 [visualization/](visualization/README.md)。模型、Loss、构造、搜索、合法化独立归各实验。官方资料与论文见 [references/](references/README.md)，未修改上游在 third_party；官方代码 Apache-2.0，数据 CC BY 4.0，保留原署名。

本地 `main` 与 GitHub [Andy-yubao/iccad2026-problem-c](https://github.com/Andy-yubao/iccad2026-problem-c) 的 `origin/main` 管理源码和必要证据。数据、环境、缓存、checkpoint 与凭据不进入 Git；E10 运行依赖本机保留的冻结权重，详见实验 README。[Git 约定](docs/git.md) · [质量目标](docs/aim.md) · [题目全文](references/official/problem_statement.md)。
