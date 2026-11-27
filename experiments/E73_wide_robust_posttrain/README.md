# E73：保留图聚合映射的宽模型后训练

2026-10-29已完成：28轮正常早停/best20，dev固定Loss仅下降2.04%，train/dev差距46.15%；交面积7.888%、flash加权Q1.033736未胜E42。同100例重叠/外框/空白均退化，新增T的下降主要来自非相交模块对。保留证据，不升级，详细[完成复核](runs/inspection.md)；下一组为E74的碰撞门控Loss＋冻结32k组合。

2026-10-29，继续用户授权的长期研究。兼容思路优先组合；保留192×12扩大模型和固定拓扑Loss，从同一个E42 best51函数保持暖启，冻结已学好的图聚合Linear，同时对其余可训练模块施加实际Adam步长限制。不是从失稳的E72 latest继续，也不扩大数据、改变Loss或重新初始化整网。

## 失败分析先于方案

E72在第5575步触发有限值爆炸保护。固定train/dev Loss从epoch0的0.583751/0.751154变为第11轮0.861870/0.904268，均劣于起点。没有“训练变好、开发变差”的证据，因此目前先修优化稳定性，不先扩32k，也不能声称已经完成容量收益验证。

[同一被拒训练批次诊断](runs/failure_diagnosis/rejected_batch.json)仅复现一个32例批次，无参数更新；它不是被拒Dropout掩码的精确回放，因为故障checkpoint的RNG已在被拒重放之后。eval下，初始/失稳前权重的批均Loss为0.404177/4.733777，梯度范数5.2867/1247.4668，隐藏最大31.39/403.15。异常主要集中在21模块成员train9065：Loss118.89，其中原G111.37、contact190.70，新增拓扑项加权仅7.52。它不是大模块数案例，也不是新拓扑项独自异常。

[五例干预](runs/failure_diagnosis/graph_interventions.json)发现：保持其他失稳参数，只恢复图聚合Linear到暖启值，train9065的Loss由118.89降至0.4279，全五例隐藏最大降至60.35。直接把图分支关闭虽也降低该例，但另外的train15603/8658升至18.32/6.44；不能将“关闭图”当作普遍修复。失稳权重的各层图Linear矩阵相对起点漂移约99%～130%，说明原单步1%保护没有约束长程累计漂移。

这些后验干预支持图路径参与放大，不是完整训练因果消融、首个跨界参数的定位或全分布稳定证明。Net2Wider只保持初始eval函数，不保持Adam轨迹：复制后某些输入列权重减半，固定绝对Adam步长的相对影响会变大；这一优化解释尚未被单独证明。

## 本组固定设计

- 同192×12模型和均匀二倍Net2Wider映射，保留原训练的E42图聚合前向值；全程固定每层`local` Linear的weight和bias，不删除图分支，不加图输入归一化。
- 其余attention、FFN、输入、head、LayerNorm继续学习。每个可训练模块的AdamW提议后，weight/bias联合位移不超过其更新前联合范数的0.5%；这是固定优化约束，所有LR档位相同，不按结果调阈值。固定源AdamW四档LR仍为1e-3/2.5e-4/1e-4/5e-5。
- 唯一Loss仍是E72的`G + 0.5 T`，同一公式、权重与聚合用于反传、固定train/dev500和完整开发评价。best/LR/早停只用dev500同Loss，epoch0仍合格，Q仅终末辅助。
- 同v5 train16000、固定train/dev500、seed0、batch32、FP32、正常四档调度和早停；Adam/调度器重置为新后训练阶段。E72的12.33分钟失败预算保留，E73新阶段最多2小时/200轮，不混称精确续接。

图模块固定后仍有更多可训练参数，但总容量与有效可训练容量须分别报告。组合同时改变了优化参数域和实际步长，不能将收益独归宽度、固定图分支或新监督。若此组健康但train/dev固定Loss均无法改善，应进一步检验表达/目标，而不是把小泛化间隙当成功；若train显著改善、dev明显退化，再考虑冻结嵌套32k；若Loss下降而Q上升，优先检查目标对齐和修复反例。

## 启动、停止与恢复

```powershell
.conda/envs/iccad2026/python.exe -u -B -m experiments.E73_wide_robust_posttrain.preflight
.conda/envs/iccad2026/python.exe -u -B -m experiments.E73_wide_robust_posttrain.pipeline --resume
```

五例预检除原固定Loss/CUDA/恢复检查外，确认图参数初始与两步后逐字相同、无图参数梯度、真实第一步各可训练组相对位移≤0.005001。`runs/preflight_v1/`保留首次检查，最终预检新增实际步长边界验证；没有用于正式训练或选模。

独立环境与单GPU；CPU终末2进程×2线程。预计训练约1～2小时，未实测本方案全程；最坏2小时预算先停止并保留完整状态。输出`runs/20261029_e42_topology_seed0/`；stage日志为`runs/train.log`、`final.log`、`draft100.log`和`report.log`。`runs/STOP`安全停止，移除STOP后同命令`--resume`，预算/RNG/调度/游标/步长计数不重置。

原pipeline训练结束后只对唯一own-Loss best做完整dev2000、一次flash500、固定100图reference/current/E42无评分对比和自动汇总。基线未升级，测试继续封存。长期总账和30分钟跟进仍在[E72 study.json](../E72_topology_posttrain/study.json)，本组不另建重复定时任务。
