# E95：独立全局修正器

当前从4481步断点迁移到加速引擎续训，用户已授权恢复新的2小时训练窗口。[实时状态](state.json) · [固定计划](plan.json) · [迁移登记](engine_migration.json) · [工程检查](runs/20261101_engine_repair/check.json)。

Loss保持`overlap + 8.46*B/A + 0.05*wirelength + 0.2*soft + 0.1*intent`。原五例梯度标定、成员、冻结E90来源、batch32和FP32均保留。F仍为192宽/12层/4头、完整公开模块与端口全局交互、一次并行输出；无内部搜索或额外调用。

新引擎采用FP32高效SDPA及32×120×480固定padding的CUDA图，捕获F/Loss/反传/裁剪；Adam前数值门禁、原AdamW和参数有限性检查保留。FP32/无AMP/无TF32，避免旧大注意力反向重算。此为数值路径迁移：并行归约和padding改变浮点舍入/Dropout消耗，不称旧路径逐位续跑。原固定形状手工/新注意力前向差约1.7e-6、梯度相对差6.4e-7，真实Adam一步差7.3e-6；新引擎保存恢复下一步模型/Adam/RNG逐位一致。

同最大尺寸四步微基准613→350ms/步，约1.75倍；五例较小自然形状271ms/步，固定padding会浪费计算，因此不能据微基准宣称完整训练同倍加速。正式吞吐看日志。公开train批次端口中位最大数419，最大475，480保留全部端口。

新目录`runs/20261101_bbox_engine_resume`加载原latest的模型/Adam/RNG/排列/游标/seen/调度/历史与best，不重新初始化；原latest/best与STOP保留。继承best在同dev500新数值路径复核，误差限2e-5。旧工程阶段训练2558.50秒加更早7200.69秒继续留账；新增7200秒窗口，原9872步匹配上限由本次明确授权替代，保留累计200轮/100000步及原早停。准备费用另记；评价新窗口1小时。

唯一own-Loss best按原计划完成dev2000草稿、dev500一次flash和同100例reference/L/F×1/后处理图；2/4/8次仅辅助。保持原始分片与冻结索引，测试封存，无评分控件/自动开图/额外候选/自动唤醒。

首次迁移入口（后台已运行后不要重复启动）：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E95_single_pass_corrector.pipeline --migrate-from experiments/E95_single_pass_corrector/runs/20261101_direct_bbox846_seed0
```

正常中断用同模块`pipeline --resume`，只移除当前新运行与实验STOP，旧运行STOP保持。旧源码复现须使用各run/source_snapshot.zip。

旧结果：[归一化修复评价](runs/20261101_normalized_local_seed0/evaluation/summary.json)（消碰明显、外框外扩、Q未升级）· [首轮数值诊断](runs/20261101_diagnosis/report.md)。
