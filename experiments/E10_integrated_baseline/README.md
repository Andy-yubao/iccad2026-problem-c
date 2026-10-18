# E10：E7 模型与 E8 后处理联合基线

2026-10-18 整合。当前工作的唯一学习型基线：冻结 E7 `attention_geometry` 第77轮 best 模型，接 E8 `conic_robust` 后处理。E9 中心修正候选在另一开发100例退化，不纳入。整合不新增训练、不改变算法或冻结数据划分。

## 使用

```python
from experiments.E10_integrated_baseline.pipeline import Baseline
solver = Baseline(device='cpu')  # 可显式使用 cuda
positions, stats = solver.solve(masked_solver_inputs)
```

模型加载一次；输入是 datasets 提供的公开、已遮罩问题元组，不能传自由块参考位置或评分。单独后处理入口是 [optimizer.py](optimizer.py)。实现独立保存在本实验，不导入其他实验的私有算法。模型和几何源文件从 E8 的已验证版本按依赖复制，来源与原文件哈希见[迁移清单](../../history/20261018_E10_consolidation/manifest.json)。

默认后处理总目标5秒，普通路径1.3秒，无有效改善时才启用可行种子救援；求解器返回和严格检查可能使实际时间超过目标，不把它描述成硬实时保证。推理时间另计，加载、数据读取和写盘不计入 stats 的端到端时间。

本地权重：`runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt`，SHA-256 `47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28`。best/latest 与原训练记录整组迁入，只留一份；历史记录的命令、路径和哈希原样保留。权重不进入 Git；远程代码克隆后须从本机或独立备份恢复上述权重，入口会核对哈希。源码、训练历史与恢复预算可追溯不代表已授权续训。

## 继承的证据与限制

| 已完成开发面板 | 平均中性时间Q | 最大Q | 硬合法 |
|---|---:|---:|---:|
| dev_quality100 | 1.211556 | 1.472866 | 100/100 |
| 额外开发100例 | 1.232217 | 2.047516 | 100/100 |

这些是 E8 原实验结果，整合不重新生成200例成绩，也不宣称质量提升。相对各自旧参照的推理＋后处理加速为8.66/10.14倍，原计时和复用边界见[原dev100报告](../../history/20261018_E10_consolidation/experiments/E8_fast_robust_postprocess/runs/20261018_robust_dev100_seed0/research.md)及[另一组开发100例报告](../../history/20261018_E10_consolidation/experiments/E8_fast_robust_postprocess/runs/20261018_robust_confirmation100_seed0/research.md)。两组已参与开发与修订，不是独立测试；平均Q≤1.20与最大Q≤1.8尚未全面达到。Q不是正式比赛分数。

E7历史已访问的部分测试影响过研究判断，默认测试继续封存，不恢复中断任务。后续兼容面板规则仍须按[协议](../../docs/protocol.md)单独批准。

## 轻量验证

迁移检查只取冻结 dev_quality 的 worker 0、24、49、74、99 各1例，保存官方评分与独立硬检查。新目录必需，已有运行拒绝覆盖：

```powershell
& .\.conda\envs\iccad2026\python.exe -B experiments/E10_integrated_baseline/run.py --output experiments/E10_integrated_baseline/runs/20261018_migration_dev5_seed0
```

本轮实际[验证记录](runs/20261018_migration_dev5_seed0/results.json)为5/5硬合法，只检验整合后的可用性；[检查说明](runs/20261018_migration_dev5_seed0/analysis.md)及[静态检查](runs/20261018_migration_dev5_seed0/workspace.json)核验当前入口、官方原件与冻结划分。[源码来源](provenance.json)及完整历史与恢复映射见[归档入口](../../history/20261018_E10_consolidation/README.md)。后续从本目录继续，新机制另建用户批准的 `En_xxx` 实验。
