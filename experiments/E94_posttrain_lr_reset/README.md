# E94：后训练学习率重置到1e-3

用户2026-11-01要求新的后训练从1e-3开始、训练后自动验证500例。前一句0.01与最后具体要求不一致，按明确的1e-3执行。假设是E93较低后训练LR限制优化，不能事前认定为失败原因。

复用E92 P best36模型及与其匹配的完整AdamW矩/固有step36000、步长保护状态，使用P terminal39随机流及采样epoch39，与E93 C保持相同初始化。唯一训练候选F，Loss精确沿用E93 C/E90：reference=1、overlap=0.08975162861857286、task=0.3336832108335229，表达式/归一化/算术均值不变。192×12、5,829,939全参数、32k整轮、batch32、FP32、dropout0.05、AdamW decay0.0003、clip1、每模块联合步长0.005不变，不复用E91/E93后训练权重。

后训练固定四档1e-3→2.5e-4→1e-4→5e-5。epoch0作新基准，无保护期；连续3轮无明显改善降档，min_delta=max(1e-5,|基准|×0.001)。小改善保存best但不清零停滞，早停连续计数不因降档清零；完成3次下降、最低档至少完成2轮、连续无明显改善≥3才早停。训练、train500/dev500监测、best、LR和早停共用同一固定F Loss，不按Q选模。

累计训练上限2h或64新增完整epoch，允许早停。准备预检另计，历史费用保留。相比E93 C，改变整套LR日程及预算上限，不能单独归因于初始LR或声称实际更新量相等。记录高LR实际步长裁限，不放宽0.005保护。逐例及均值Loss<1000、有限值、裁剪前梯度范数<10000均在Adam前检查；异常停止、不跳批、不自动重试。阶段内恢复完整保留Adam/RNG/排列/游标/调度和预算。

启动前最多5个训练成员检查真实高LR更新、与1e-4控制的差异、精确中断恢复、调度和数值保护。入口：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E94_posttrain_lr_reset.preflight
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E94_posttrain_lr_reset.pipeline
```

训练正常早停/预算结束后，仅冻结F自身Loss的唯一best，复现dev500 Loss，再以原E90相同flash参数、4进程×2线程执行一次固定500例开发评价，输出覆盖/尾部/外框/空白/耗时/约束及固定100张reference/E94/E90草稿图。E90/E92/E93 C/G旧结果只读复用，不额外跑dev2000，不按终末Q反选或自动再训练。评价另限累计1h，只续同模型未完成成员。测试封存，无修正研究或自动唤醒。

输出：`runs/20261101_lr1e3_seed0/`；[状态](state.json)、[启动记录](runs/launch.json)、`runs/preflight/results.json`。验证及汇总为该run的`qualification/report.md`，来源及累计费用见`provenance.json`和`runs/authorization.json`。算法由E93独立复制，不跨实验导入私有算法。

创建`runs/STOP`安全停止；显式恢复使用pipeline加`--resume`，累计费用不清零，已完成部分不重跑。错误运行必须先诊断登记。训练预计不超过2h，自动评价预估5–15分钟，未作本次全量实测；训练使用GPU及4个CPU线程，终末4进程×2线程。
