# E71：温和reference日程＋内部密度＋紧凑约束

2026-10-28训练与终末评价已完成：[成果复核](runs/inspection.md)。70轮正常早停／best67，dev2000交面积4.972%，相对E70外框缩小但重叠回升；同100例较E42外框平均小0.647%、成对交面积低约20.43%。flash500全硬合法但加权Q1.038751及尾部未胜出。保留草稿候选，不升级E42，不续训，测试封存。下文为训练前准备与复现说明。

2026-10-28用户要求采用较快开发节奏，将早期reference增强与内部协调合为一个seed0候选，不拆分训练消融。准备阶段仅交代码、最多五例短查和运行命令；随后由用户执行正式训练，当前已完成。原E70、E68及E42权重保留，测试封存。

## 设计与比较

从E70独立复制运行路径，保持96×12、端口宽24的完整端口结构（1,532,883参数）、v5 train16000、固定train/dev500、batch32、FP32和从头seed0初始化。组合修改：位置权重第1～5轮0.375，第6～15轮余弦降至0.25，之后固定；形状保持0.25；紧凑项0.25提高至0.5；新增权重0.5的4/8阶多尺度解析矩形密度项。其余八项保留E70定义，reference仍仅用于位置/形状。具体公式、局限、容量判断见[loss_design.md](loss_design.md)，固定配置见[proposal.json](proposal.json)、[执行计划](plan.json)、[输入边界](input_audit.md)。

这是明确登记的固定权重规则局部例外。训练反传与当期train/dev监测使用相同公式和日程；第16轮起才保存正式best并启动LR/早停判定，之后三者统一使用稳定的最终总Loss。早期当期Loss不跨阶段作改善结论，另存最终权重总Loss作固定诊断；旧G及Q都不选模。若资源预算结束时尚无合格best，停止并报告，不评价早期权重模型。

直接与已存E70 best71、E42等同例结果比较，不能把组合收益拆成单项因果。用户草稿观感优先；同时记录实际交面积、外框/A、精确矩形并集与空白、原始线长、无端口位置/近邻诊断、弱端口稀疏子群、硬合法性与flash尾部。唯一best完成dev2000、flash500及原同100例reference/current/E42无评分图；不自动升级基线。

本轮不扩大模型和数据。这是基于当前目标缺口和既有表现的研究判断，不宣称容量充足已被证明。下一步若需扩模型，优先宽128×12；若需扩数据，优先16k→32k且保持开发/测试成员不变，均未准备或排队。不会自动恢复旧160宽或64k实验。

## 短查与启动

先执行最多五个唯一训练成员的实现短查，产物位于[runs/preflight/results.json](runs/preflight/results.json)：

```powershell
Set-Location 'E:\project\ICCAD\2026\problemC'
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E71_density_reference_curriculum.preflight
```

用户正式执行：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E71_density_reference_curriculum.pipeline --resume
```

预览命令用同入口加`--dry-run`。单组累计训练上限3小时/200轮，不因恢复清零；预计约2～3小时训练，终末CPU评价和100图另需数分钟至十余分钟，此估计尚未经新Loss全程测量。使用一张CUDA显卡，GPU常驻共享数据；实际显存与短步计时写入短查结果。正式输出`runs/20261028_full_ports_density_curriculum_v1_seed0/`，阶段日志为`runs/full_ports_train.log`、`full_ports_final.log`、`full_ports_draft100.log`，汇总`runs/review.md`。

Ctrl+C或创建本实验`runs/STOP`会在保存边界停止；恢复时移除自己创建的STOP后重用上述`--resume`。best/latest分别保存选模模型和包含优化器、调度、样本游标、RNG与累计预算的可恢复状态。源码修改后须重新短查；已开始运行拒绝源码/数据身份漂移。本候选结束即停止，不自动续跑、扩容、扩数据、开启评分界面或定时任务。

本轮短计时时电脑为电池供电，GPU满负载约180MHz、P8并触发软件功率限制；启用/禁用密度项分别约1.529/1.522秒一步，增量约0.5%，不能将该绝对速度用于正常供电的耗时估计。证据见[runs/runtime_probe.json](runs/runtime_probe.json)。正式启动前请接上电源；上述2～3小时为正常供电下的初步估计，不保证本机当前状态可达到。短查小批峰值显存约3.28GiB，不代表完整常驻16k数据的总显存。
