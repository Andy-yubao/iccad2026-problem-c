# E66：冻结G下的增量输出与共享迭代修正

两组均已正常早停并完成raw2000、flash500及各100图。unshared/shared dev500 G为0.379448/0.381916，较E64高3.04%/3.71%，固定train G亦高10.86%/12.56%；差距缩小伴随拟合变差，未证实泛化改善。已抽看5张既有同例图、核对保护权重，不再扩展当前配置，保留证据；下一阶段独立E67。见[完成复核](runs/review.md)，测试封存。

两组五例检查均通过，已后台启动unshared_residual seed0并核对实际更新；控制器随后依次评价唯一best并运行shared_residual seed0。固定G／梯度与实际CUDA/eager Adam更新核对通过，Dropout下断点恢复逐位一致；参数量分别1,481,334与386,610，有效块应用均12次。实时进度以state及运行日志为准。

2026-10-28长期任务第二阶段，用户授权助手自行后台训练和每小时继续，明确不做多种子，全部seed0。E65两组已完成，几何组相对同容量对照有小幅开发/重叠收益，保留候选但不追加E65配置。E66独立使用原E64主干，不叠加几何消息。[本阶段状态](state.json) · [总体计划](../E65_geometry_messages/long_task.md) · [累计预算](../E65_geometry_messages/long_task.json)。

## 原理与比较

公开节点特征先得到初始state，每一步根据当前矩形重新构造原矩形注意力，单独的归一化MLP给出delta。用`state += mask * delta_bounded`更新，最后只计算原八项G，不增加中间监督、不用reference初始化或扰动。

每步中心最大位移为`0.5*max(sqrt(归一化公开面积),0.1)`，log宽高比增量上限0.5，均用tanh平滑限幅。公开全局尺度下的0.1下限避免小模块因步幅过小无法跨区域移动。所有固定/预放置及padding mask保持原样；最终原decode不变。这是新的表示偏置，不能预先认定比原直接输出更好。

| 组 | 独立矩形块 | 应用次数 | 目的 |
|---|---:|---:|---|
| unshared_residual | 12 | 12 | 与E64比较增量修正这一组合表示 |
| shared_residual | 3 | 每块4次，共12 | 与上组比较共享算子/参数约束 |

两组都先按原随机顺序初始化完整12块，再保留12或前3块，故初始coarse head、前3块及随机状态对齐；另加的correction head在fork_rng内相同初始化，最后一层权重和bias为零。初始输出两组相同，等于coarse head，不等于原E64深层随机输出。主head只给初始state，新correction head在12步间共享。Dropout在每次块应用重新抽样，重复块不是重复同一个Dropout mask。

E64到unshared同时改变增量输出、步幅与readout归一化，不能把差异归于某一个小组件。E66两组有效深度相同、唯一参数量不同；报告训练/开发同口径G、共同更新数轨迹、各自best、参数与耗时。共享可能抑制过拟合，也可能欠拟合，不能只看train—dev差距变小。

## 冻结与执行

v5 train16000、固定train/dev500、原顺序；冻结E64八项G，训练/监测/选模/LR/早停同一Loss，仅最终输出。batch32、FP32、clip1、AdamW decay3e-4、FFN dropout.05、原四档LR及15轮保护期、最低档至少两完整轮，均保持。每组最多3小时/200完整轮，失败/恢复计入全任务18小时/10次上限，72小时日历截止不重置；本阶段只有上述两组seed0。

同五train案例检查：Loss/梯度、12次有效应用、步幅与mask、零初始化、CUDA/eager实际Adam、非有限值阻止更新以及Dropout断点恢复逐位一致。检查通过且源码身份相同才启动。来源：[provenance.json](provenance.json)，精确预登记：[plan.json](plan.json)，模型：[model.py](model.py)，冻结主干：[base_model.py](base_model.py)。不跨实验导入私有算法，仅共享总体预算JSON数据。

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E66_shared_refinement.preflight
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E66_shared_refinement.pipeline --resume
```

项目根执行，用`.conda/envs/iccad2026`。预检证据：[runs/20261028_preflight/results.json](runs/20261028_preflight/results.json)。持续exec会话保持父控制器；不得Start-Process后立即退出父shell。控制器依次完成unshared的训练/唯一best raw2000/flash500/100图，再做shared。Q仅辅助，不按Q重选、不自动升级；用户草稿图评优先，测试封存。

输出`runs/20261028_<arm>_seed0/`，日志`runs/<arm>_train.log`，完成报告[runs/review.md](runs/review.md)与[review.json](runs/review.json)。CPU评价2进程各2线程、图4进程各1线程，总4线程。停止：创建本实验`runs/STOP`；同原代码`pipeline --resume`恢复，先移除自己创建的STOP，保留优化器、调度、RNG、游标与累计预算，不从头重跑或延长预算。原E42/E64/E65 checkpoint开始/结束核对保护哈希。
