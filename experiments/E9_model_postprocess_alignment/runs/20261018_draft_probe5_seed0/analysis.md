# E9 第一阶段：草稿诊断与目标可达性

2026-10-18。实际完成固定 dev_quality 中 worker 0/24/49/74/99 各一例，共 5 例、25 次求解。全部使用 E9 内逐字复制的 E8 `conic_robust`，普通例 1.3 秒、整例目标 5 秒；全部 25 个输出通过官方及独立硬检查。整次诊断墙钟 70.855 秒，求解均值约 1.24–1.26 秒/例。没有训练、没有读取封存测试，也没有扩大到额外开发实例推理。

## 核心发现

**固定 dev100 上的原定平均 Q＜1.10 目标不可达，原因是公开 MIB 规格与面积/锁定规格之间存在不可消除的冲突。** [完整下界推导](../20261018_official_quality_bound_dev100_seed0_r2/analysis.md)从既有保存的 100 例公开输入规格推导出算术平均 Q 的保守下界 **1.180663**。下界计入官方四位小数同形判定、固定规格容差、面积容差及 Q 封顶；不是把当前输出的 MIB 违反直接认作最小值。

当前 E8 的同面板均值为 1.211556，距该乐观下界约 0.030893。下界忽略其他质量损失，不能保证模型一定能做到 1.180663，不能据此保证 1.19 可达。该结论仅适用于当前固定本地面板和计分规则，不推广成隐藏集结论。

以 worker0 的一组 MIB 为例，其目标面积分别是 72、240、56，每个软块仅能在 1% 内变化；普通尺寸下不能共享宽高。即使使用原参考草稿，组内仍为 3 种尺寸、产生 2 次 MIB 违反。并非训练没拟合好，也不是锥求解的细微数值误差导致这两次违反。

[两组既有结果分解](../20261018_saved_quality_audit_seed0/analysis.md)显示，MIB 占 dev100/confirm100 平均 log Q 损失的 90.6%/83.2%。这只是得分分解；“在相同坐标上去掉 MIB 罚项”的反事实不代表能合法做到。主评分、约束和冻结数据均未改变。

## 同实例受控对照

| 草稿输入 | 后处理后的平均 Q | 平均 MIB 违反 | 硬合法 |
|---|---:|---:|---:|
| 原 E7 草稿 | 1.151565 | 3.6 | 5/5 |
| 只按公开 MIB 对兼容尺寸协调 | 1.151565 | 3.6 | 5/5 |
| 原中心＋参考形状 | 1.148257 | 3.6 | 5/5 |
| 参考中心＋原形状 | 1.131604 | 3.6 | 5/5 |
| 完整参考草稿 | 1.131070 | 3.6 | 5/5 |

原参考布局本身平均 Q 为 1.245554，包含边界/MIB 违反；不能将它当成 Q=1 的标签。参考形状、参考中心与完整参考都是**诊断用 oracle**，不能部署，也不是模型实现的成绩。形状/中心替换后保留公开 fixed/preplaced 规格，非负坐标裁切可能引入附带中心变化；这些副作用包含在对照定义中，不将差异归结为纯粹单一几何量。

本五例中，参考中心提供的信号大于单独参考形状。公开尺寸协调几乎不改变最终结果，说明简单给草稿增加一次 MIB 协调不能解决规格本身的不兼容。该现象支持后续优先研究全局位置组织与公开约束协调，不支持继续加大不可能同形组的连续形状惩罚，也不足以断言新模型在完整 dev100 上必然改善。

## 下界核验

[独立证书检查](../20261018_bound_verification_seed0/results.json)通过：100 例、100 个 MIB 组、513 个面积区间；覆盖点数与两两不交区间证书一致。既有 100 个 E8 输出和本次 25 个输出均不违反下界，五例实际公开约束与 MIB 面积也与既有记录相同。另构造了极端长宽比下“不同面积仍落在同一四位小数形状桶”的反例，确认不能偷用不考虑舍入的简单面积下界。本次保守证明已处理此反例。

对应固定 100 例，现有 E8 的 MIB 违反总数为 393，保守不可避免下界为 374。其余 19 次并不保证全部可消除；宽高、分组、边界和线长还有耦合。

## 实施取舍与下一步

没有启动正式训练，也没有用五例诊断选择训练超参数。原目标可达性被推翻后，先明确后续目标更合理：保留官方 Q，先争取 dev100＜1.20、再探索 1.19；能否达到仍要实测，confirm100 继续单列。若继续模型研究，优先 A 的质量目标对齐和 B 的全局组织能力，保留同预算原 geometry 控制组；不重新引入已停止的 occupancy/rectangle 方向。

目前新目标已向用户提出，尚待确认；未把建议写成已获批准的新训练目标。本轮仅实现并完成诊断入口，**尚未实现新模型或交付正式训练命令**。

## 复现与图集

[浏览五例对照图](index.html) · [实际配置、命令与来源哈希](config.json) · [逐例输出与 MIB 尺寸](cases.jsonl) · [完整统计](results.json) · [可恢复源码](source_snapshot.zip)。布局图统一调用 `visualization.layouts`，只读取已保存坐标，不再加载数据或求解。

从工作区根目录使用独立环境执行下列命令。输出目录均为新的复现目录，不覆盖本次结果。诊断预计 1–3 分钟、CPU 求解、不使用 GPU 模型前向，进程可能加载 PyTorch CUDA 库但不做 CUDA 计算；输出约数 MB。`--budget-seconds 600` 限制累计墙钟，单次求解另有 15 秒子进程保护。创建该输出目录的 `STOP` 会在当前子问题返回后停止并保存；本轻量诊断入口不提供续跑，若强制中断或需要重跑，另开输出目录。

```powershell
Set-Location E:\project\ICCAD\2026\problemC
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.analyze_saved --output experiments/E9_model_postprocess_alignment/runs/20261018_saved_quality_audit_seed0_reproduce
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.probe --output experiments/E9_model_postprocess_alignment/runs/20261018_draft_probe5_seed0_reproduce --budget-seconds 600
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.quality_bound --output experiments/E9_model_postprocess_alignment/runs/20261018_official_quality_bound_dev100_seed0_reproduce
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.verify_bound --bound experiments/E9_model_postprocess_alignment/runs/20261018_official_quality_bound_dev100_seed0_reproduce --probe experiments/E9_model_postprocess_alignment/runs/20261018_draft_probe5_seed0_reproduce --output experiments/E9_model_postprocess_alignment/runs/20261018_bound_verification_seed0_reproduce
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.render_probe --run experiments/E9_model_postprocess_alignment/runs/20261018_draft_probe5_seed0_reproduce
```

首次下界运行在源码快照阶段因记录字段重名失败，未读取张量或计算下界；[失败事实](../20261018_official_quality_bound_dev100_seed0/results.json)保留。修复后新建 r2 完整运行，没有覆盖原记录或挑选统计结果。
