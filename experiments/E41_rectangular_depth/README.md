# E41：E39矩形注意力继续扩深

2026-10-23用户自行运行后已完成：45轮早停、best42、完整开发代理0.479912，较E39矩形12层0.468025回退2.54%；终末500全硬合法，加权Q1.030685，最大Q与最差5%有所改善。[完成结果与回退分析](runs/20261023_rectangle_w96d16_seed0/analysis.md)已核对配置/源码/数据与共同轮数，未续训或重评。定时任务不恢复。下文保留原准备方案与命令，不构成新训练安排。

MIB平均面积处理继续保留。该处理使同形要求在面积上可兼容，监督冲突并不是撤销这一处理的理由。本轮只增加深度，MIB Loss系数仍为0.1；增加系数、共享形状解码、监督修订和预放置专用机制均留作后续独立工作，不混入本轮。

## 实验与判断

唯一新训练：**rectangle，宽96、16层、seed0，从头初始化，1,958,083参数**。参照只读复用[E39 rectangle 96×12 seed0 best49](../E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0/analysis.md)，不重复训练12层或评价E25；没有第二个排队训练。

假设：在矩形间隙/穿透信息不变时，更多消息传递轮次能够进一步降低固定开发代理及碰撞严重度，也可能改善障碍附近的相对关系。反例包括：只降低训练Loss、验证不降；碰撞改善但边界/MIB/接触或尾部退化；增深带来的推理成本抵消修复收益。一次seed0结果不证明稳定收益，更不证明监督冲突被解决。

保持原v5 train16000/dev2000、固定train/dev_monitor各500、MIB面积变换、原参考监督、训练Loss和解码、后处理不变。0.468025是12层完整dev2000的geometry＋occupancy代理；对应geometry=0.365913。500监测代理0.471997，不能与2000例数字混比。无固定Q/耗时通过线，不根据终末Q更换best。

E39代码原入口只支持4/8/12层，其已完成结果绑定源码身份；因此本次独立复制至E41，保护E39冻结源码。不是新矩形机制，唯一模型配置变化是12→16层。[provenance.json](provenance.json)记录44个复制文件的原始/本地哈希；model、Loss、特征、批处理、缓存语义与求解器保持字节一致。运行命名、环境变量前缀、修订号与输出归属使用E41；训练预算、调度和恢复算法沿用验证过的E39 v2。

## 运行命令

在PowerShell粘贴以下命令。脚本自行切换工作区，使用本地独立环境；**现在不要运行，等训练条件具备后再由用户执行。**

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E41_rectangular_depth\launch.ps1' -Stage all -Resume
```

`-Resume`首次运行时从头初始化；已有latest时严格核对源码/数据/设备/累计预算后恢复。若已结束，复用完成结果。执行顺序只有：

1. 新训16层，训练/监测不调用求解器。
2. 冻结唯一best，完整dev2000原始评价。
3. 一次固定dev500终末后处理评价，CPU计时与GPU训练分时。
4. 只读汇总12层/16层的总体及固定难例分层，结束，不再开启任何训练。

仅训练而不接续评价：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E41_rectangular_depth\launch.ps1' -Stage train -Resume
```

已完成训练后单独评价同一best：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E41_rectangular_depth\launch.ps1' -Stage final -Resume
```

只打印命令而不训练/评价/创建正式运行目录：加`-DryRun`。如PowerShell执行策略拦截，用户可在已检查脚本后通过其既有允许方式运行，不需要修改全局策略。

## 预算、资源与停止恢复

- 单GPU CUDA，batch32，不自动降低batch、不改确定性；CPU总计四线程，终末2进程×2线程。
- 累计训练阶段最多4小时或200轮，沿用best/latest、优化器/调度器/随机状态/成员游标完整恢复，启动、数据载入及固定监测计入预算；中断不清零。
- LR固定`1e-3 → 2.5e-4 → 1e-4 → 5e-5`。第15轮起连续3轮无明显改善降档；下降3次且最低档完成至少两轮后才允许早停，沿用已有改善阈值和前两轮物理项渐入。
- 12层此前50轮约86分钟；按层数粗估16层相近轮数约2小时，规划留2～4小时窗口。**不是16层实测速度/收敛轮数或显存保证**；新配置尚未进行CUDA训练入口或显存测试。显存不足时报错保存证据，不自动更改配置。
- 完整开发与唯一终末评价另计，预计几分钟，未对16层实测。

安全停止可按Ctrl+C，或在另一个PowerShell执行：

```powershell
New-Item -ItemType File -Force -Path 'E:\project\ICCAD\2026\problemC\experiments\E41_rectangular_depth\runs\STOP' | Out-Null
```

在当前batch/病例边界保存退出。确认停止后，恢复前只移除本实验STOP，再重用同一启动命令：

```powershell
Remove-Item -LiteralPath 'E:\project\ICCAD\2026\problemC\experiments\E41_rectangular_depth\runs\STOP' -ErrorAction SilentlyContinue
& 'E:\project\ICCAD\2026\problemC\experiments\E41_rectangular_depth\launch.ps1' -Stage all -Resume
```

如自行另建了模型目录或pipeline目录的STOP，也须在明确恢复时移除对应文件。已有结束状态不会重新训练；预算结束不作为续加4小时的入口。

## 输出与分层诊断

- 模型目录：`runs/20261023_rectangle_w96d16_seed0/`。
- 协调日志：`runs/20261023_rectangle_w96d16_pipeline_seed0/console.log`；同目录active/progress保存阶段、命令和停止事实。
- best/latest在模型目录`checkpoints/`，`progress.json`保存每轮固定训练/验证及分项，`full_dev.json`为完整原始开发评价，`terminal/results.json`保存唯一终末500坐标与轨迹。
- `comparison.json`/`comparison.md`将与冻结12层配对：总体、输入预放置占比最高四分位、**按旧E39冻结的**重叠最高四分位、25例旧早期投影、6例旧救援及12例group≥50%子集。
- 旧模型条件化难例存在均值回归，6/12/25例小子集只作机制诊断，不作为单独选模/通过门槛。分层不改变全体500分母，不用候选结果重新选难例。
- 主要看完整开发geometry/occupancy分项、碰撞严重度、train/dev差距；另列边界/group/MIB、合法率、位移、投影/救援、Q尾部及完整计时。一次限时修复与单种子不支持稳定部署升级结论。

参考[失败模式审计](../E39_rectangular_attention/runs/20261023_failure_mode_audit/analysis.md)。不新增教师、不访问测试、不修改E25/baseline，不恢复已取消的训练队列或定时任务。

## 已做与未做的检查

[准备检查](runs/20261023_preparation/results.json)仅CPU、一例既有开发成员：16层配置和实际参数量、有限前向、Loss权重不变；将旧12层checkpoint严格加载到复制实现后，输出与原E39逐元素一致。检查没有优化器更新、CUDA执行或求解器调用。PowerShell语法与`-DryRun`实际入口核对另存同目录。

训练及GPU显存/吞吐尚未实测；未创建正式模型/协调运行目录。[命令入口](launch.ps1)、[顺序协调器](pipeline.py)、[只读结果汇总](report.py)。
