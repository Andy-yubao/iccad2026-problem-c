# 启动、停止与结果

在仓库根目录的PowerShell执行：

```powershell
& .\experiments\E13_solver_distillation\launch.ps1 -Stage All -Arm Both
```

顺序为：完整train20,000教师标签 → 固定dev_monitor500教师标签 → 几何蒸馏及评价 → 几何+结构蒸馏及评价。教师库预计数小时；两组训练各最多4小时/200轮，通常会因验证早停提前结束。教师库分别有12小时/1小时累计保护上限。整个流程由用户启动，助手没有前台或后台启动这些长任务。

前置检查在本机RTX5080 Laptop GPU、PyTorch2.7.1+cu128下完成。正式训练采用CUDA，标签生成主要使用CPU后处理。训练预先将20,000例公开图特征和教师目标放入CPU内存，会占用数GB内存；不复制原始数据张量。每个环节失败或被STOP停止后，脚本停止向后执行，保留已完成内容。

只生成标签或分阶段运行：

```powershell
& .\experiments\E13_solver_distillation\launch.ps1 -Stage Teachers
& .\experiments\E13_solver_distillation\launch.ps1 -Stage Train -Arm Both
& .\experiments\E13_solver_distillation\launch.ps1 -Stage Evaluate -Arm Both
```

`-Arm structure`只运行结构蒸馏配置；默认Both保留已固定的几何对照。`-DryRun`只打印实际命令，不运行计算。更改RunName会创建新的训练实验，但复用同一冻结教师库；不要用换名字冒充同一训练的恢复。

## 停止与恢复

训练或标签目录下新建名为`STOP`的空文件，会在当前样例/训练batch结束后保存停止。Ctrl+C在训练阶段也会请求保存；每100个batch自动保存latest。当前完整监测环节完成后才检查时间/STOP，故时间上限可能多出一个监测周期。生成标签时请优先用STOP以保证最新一例完整保存。

例如停止当前几何训练：

```powershell
New-Item -ItemType File -Path experiments/E13_solver_distillation/runs/distill_v1_seed0/geometry/STOP
```

移除相应STOP后，重新执行同一启动命令即可自动恢复；保存的累计预算、batch排列、游标、随机数和优化器状态均沿用。正式训练不接受pilot或部分教师库，也不自动跳过失败标签。已完成阶段会跳过已保存工作，预算不会重置。若上限已耗尽，将保留停止结论；需要增加预算时应另行明确版本/预算变更，不能反复重启绕过上限。

## 输出

| 目录/文件 | 内容 |
|---|---|
| E12/runs/teacher_train20000_v1 | 冻结train教师库；labels/*.npz只保留本地 |
| E12/runs/teacher_devmonitor500_v1 | 仅评价用的开发监测教师库 |
| E13/runs/distill_v1_seed0/geometry | 仅几何蒸馏 |
| E13/runs/distill_v1_seed0/structure | 几何+结构蒸馏 |
| 各训练目录/progress.json | 固定500例监测曲线、已见成员数、学习率、早停与预算 |
| 各训练目录/label_quality.json | 同输入下教师/reference的官方中性时间Q、改善或退化标记 |
| 各训练目录/checkpoints/best.pt、latest.pt | 最佳权重与完整恢复状态；不入Git |
| 各训练目录/eval_supervised/results.json | 完整dev2500对原reference的预测误差，属于兼容性诊断 |
| 各训练目录/eval_repair/results.json | 固定dev100真实学生+轻量后处理的合法性/Q/回退/总时间 |

实际蒸馏效果以完整dev教师误差和真实学生dev100为准。五例实现检查、理想头输出、oracle结构/几何以及预留网络时间都不构成真实学生成绩。

如需训练后的机制诊断，可显式在评价命令中增加`--oracle structure`或`--oracle geometry`并另用输出目录。它们必须带完整dev_monitor教师库；配置与结果始终标记使用了教师信息。默认启动命令运行的是学生自己的输出。
