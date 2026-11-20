# E98多轮Loss调权：原基线重启、混合Q、训练反馈100/1000

用户2026-11-04明确授权准备后后台运行。旧pilot结果不作为初始化、方向依据或接纳证据；原文件和权重保护，不删除或重写历史。唯一初始模型为固定丰富Loss基底best58，SHA256=`1df099d624797d3e61fd332b7ffafed8ebc213f73758649e2a34054b395201d1`。

## 固定边界

- 原v5 train16k用于梯度，原固定train500/dev500监测，FP32/batch32/96×12/seed0不改；公开MIB平均面积派生任务和端口完全压缩不变。仅缓存E42归档路径作显式解析迁移，原缓存身份、大小、mtime和输入哈希仍检查。
- 从16k均匀无放回冻结1000例训练反馈候选池。每轮提前用独立固定种子从池中抽100例，三支与两次重复共用同100例；轮间可重叠。全部成员、方向和顺序在观察结果前冻结。
- 每轮control/plus/minus从同一父模型开始，三支均新建AdamW（m、v、内部步数清零），保留decay.0003、clip1和模块联合更新上限.005。继承父模型RNG，三支共同样本顺序；阶段内部恢复保存全部优化器/RNG/排列游标/调度/保护计数和累计耗时。
- 每支Loss系数固定；反传、train/dev监测同一函数。**验证集dev500 Loss**选择own-Loss best并控制LR/早停。训练前epoch0监测；第1–2轮保护，第2轮末建立基准，第3轮起连续3次无明显改善降LR，因此最早第5轮末从1e-4降至5e-5。明显改善=max(1e-5,基准Loss×.001)；best保存门槛1e-5。降档后清零停滞，再满3轮停滞早停；每支最多10轮。未降档或仍改善也在10轮到上限，不冒称收敛。三支实际计算量可能不同，按同起点/同规则/同上限比较，不称等更新数对照。
- 分支反馈只在各支dev-Loss best上进行；不逐epoch求Q选checkpoint。每支同100训练成员两次完整原单线程后处理，第二次反向分支顺序。质量报告weighted_Q、mean_Q、mixed_Q=.5*(weighted_Q+mean_Q)，均为中性时间质量。独立报告硬合法、完整逐例均时/P95/最大值、250ms超限数；不混入批量吞吐。
- 候选两次mixed_Q均比control至少低.0001且全部硬合法才可接纳；多候选满足则两次均值更低者，平局保持control。无改善选择control，对应训练模型可继续，因而不保证相对父模型的Q单调下降。dev后处理Q不进入接纳；测试载荷封存。

## 方向与预算

第一至第五轮是可解释的对称探测，不预设哪一侧更好：

1. reference位置/形状 ↔ 其余公共几何任务。
2. 三项重叠 ↔ 线长/紧凑度。
3. 最坏重叠超限 ↔ 穿透/平均重叠负荷。
4. 线长 ↔ 紧凑度。
5. 边界 ↔ 分组接触。

第六至第十轮采用固定种子生成的不重复Rademacher方向。每轮从已选择系数为中心，`c± ∝ c*exp(±log(1.25)*d)`，归一化保持总系数和；零方向项在共同归一化前保持原值。方向、随机种子及各轮100例面板均保存在run/plan.json。**本次有限预算为5组设计＋5组随机，共10轮**，结束后不自动追加或唤醒。

十轮结束冻结唯一最终模型，然后做一次终末dev500完整后处理和同100例reference/L/最终图，无评分控件、不自动打开。dev在内层反复用于Loss调度，故终末dev是开发评价，不称独立未见证据。测试不读取。

## 执行与停止

运行目录：`experiments/E98_loss_weight_learning/runs/20261104_dynamic_mixedq_seed0`。登记为plan.json，必要五训练例实际更新与后处理接口短检为preflight.json；每轮源码身份、训练记录、best/latest、反馈坐标和接纳分别保存。正式状态state.json、日志stdout.log/stderr.log，完成后results.json/report.md。

准备与短检：`python -u -B -m experiments.E98_loss_weight_learning.dynamic_weight_learning.prepare`，随后`python -u -B -m experiments.E98_loss_weight_learning.dynamic_weight_learning.preflight`。后台入口为本目录launch.ps1。普通未结束运行恢复用`python -u -B -m experiments.E98_loss_weight_learning.dynamic_weight_learning.run --resume`；不自动恢复error或已完成运行，不预先测试恢复。

创建本run目录的`STOP`可安全停止：批次边界保存latest，阻止后续分支/轮次/终末评价。不触碰其他实验的停止文件或后台进程。
