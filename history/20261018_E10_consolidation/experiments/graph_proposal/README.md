# E1：图条件布局提议

2026-10-17 更新：本实验保留冻结模型与原诊断作为参照，后续研究已转入 [E6：可修复提议与意图感知修复](../intent_aware_repair/README.md)。以下阶段计划与停止事实属于 E1 记录，不表示将自动在 E1 实现 B/C/D 或续训。

## 软约束图集与机制验证

已按用户要求生成[训练/验证/已访问测试各10张图](runs/20261017_soft_constraint_gallery_seed0/index.html)，grouping同色、MIB同纹理、boundary边徽记红绿状态，第三栏仅当前后处理布局。[分析](runs/20261017_soft_constraint_gallery_seed0/analysis.md)：957例既有测试Q=3.859，固定几何移除全部软罚的评分反事实为1.694，boundary影响最大；固定5例dev将B/G局部权重2改为10，Q从4.011到3.284（4胜1平，改善18.11%），支持当前软权衡不足，但未修改默认权重。测试图片仅复用此前957例中的10例，未扩读测试成员；MIB部分违反受硬规格限制，不能全部归因权重。保留式新修复器尚未实现。

## 下一阶段：独立评价提议，再研究最小改动修复

按用户最新研究目标，参考布局作为近似最优参照；分别判断相对布局、局部可修复性与最终求解质量。设计来源、当前后处理缺陷、评价指标和执行门槛见[下一阶段计划](next_stage_plan.md)。

**阶段A已完成**：[原始提议独立评价](runs/20261016_raw_proposal_train100_dev100_seed0/analysis.md)，固定train100/dev100、各100worker，GPU推理，约67秒，无后处理/训练/test。dev中心方向保留99.34%、分离方向保留94.28%，说明粗略布局已学到；但中心误差约局部尺寸65.67%、94.62%的块涉及重叠，固定形状平移必要下界中位51.07%，全部100例超过10%。固定形状下不满足微小平移可修目标，允许形变的情形尚待检验，不能将高Q全归因旧后处理。[分布图](runs/20261016_raw_proposal_train100_dev100_seed0/proposal_diagnostics.png) · [叠加图](runs/20261016_raw_proposal_train100_dev100_seed0/proposal_overlays.png)。B及后续阶段尚未执行。

## 最新诊断：容量与后处理

用户进一步授权检查训练解与最优Q的差距。已完成[5例train容量和分阶段诊断](runs/20261016_train5_capacity_diagnosis_seed0/analysis.md)，约142秒，不读dev/test、不续训原模型。当前网络80,131参数；同结构与128维6层模型在五例上分别记忆到MSE约7.17e-5/6.85e-6，最终Q却为2.686/2.999，当前冻结模型为2.674。参考解Q=1.179，经当前pack变2.352，直接证明后处理会损害完美提议。参考解自身有软违反，不能称真最优；五例加权真最优在[1,1.179]内。五例诊断不代替完整train评估，不排除全量容量问题；原best未改，不据此重开test。[诊断图](runs/20261016_train5_capacity_diagnosis_seed0/capacity_diagnostics.png)。

## 当前结论：训练与测试已停止，进入分析

用户先停止训练，再要求测试约500例即停止。执行终止时已完成 **957例**，全部保留，覆盖39/100 worker；不补跑、不续训，也不启动计划中的dev100对照。[完整分析](runs/20261016_frozen_best_test2500_seed0/analysis.md) · [汇总](runs/20261016_frozen_best_test2500_seed0/results.json) · [训练曲线](runs/20261016_deterministic_workset_v2_seed0/training_diagnostics.png)。

冻结 best step7,250（epoch5.8），完整train20,000/dev2,500的 MSE=0.0748709/0.0750770，没有明显过拟合迹象，容量欠拟合尚未证实。部分test MSE=0.0759862，957/957硬合法、零兜底、中性时间加权Q=3.859185，平均CPU1求解0.324秒；HPWL/bbox gap=+51.48%/+80.24%。测试前缀覆盖不均衡，不代表完整2500例，不可与旧dev5直接比较。测试已访问成员不再是未见数据；没有基于测试调参或选权重。训练是用户主动停止，实际降学习率一次，未触发自动早停，不声称收敛。以下命令保留为本次训练配置记录，不自动再次执行或恢复。

## 当前训练规范：v2 工作集与验证驱动重训

用户于 2026-10-16 批准后续统一使用 [workset_v2.json](../../protocol/workset_v2.json)：train **20,000**、dev **2,500**、封存 local_test **2,500**，以及嵌套 train_monitor/dev_monitor 各 **500**、dev_quality **100**。选择规则、集合隔离与封存边界见 [执行协议](../../docs/protocol.md#11-后续训练统一工作集-v2)。原 split_v1.json、旧成员、旧模型和旧结果保留原件。以下旧短预算结论属于历史阶段。

本次 [train_supervised.py](train_supervised.py) **从 seed=0 随机初始化重训确定性模型**，不读取旧权重；网络仍是 3 层、宽度 64 的图编码器和中心/宽高比回归头。变化同时包括均衡工作集、全训练成员每轮全局洗牌、batch=16、固定验证、学习率下降与早停，不能将之后的变化归因于其中单独一项。

固定训练设置：AdamW lr=1e-3、weight_decay=1e-4、梯度范数裁剪 1.0；CPU 1 线程，CUDA。每 250 次更新和每轮末计算固定 500 例 dev_monitor 的逐布局等权 masked MSE，同时记录中心误差、对数宽高比误差与固定 train_monitor，并区分监测样本是否已经训练过。原始监督损失定义不改，不把最终 Q 混入 Loss。

- **学习率下降**：ReduceLROnPlateau，验证相对改善阈值 0.2%，patience=2 次验证、cooldown=1，倍率 0.5，最低 1e-5。
- **早停**：连续 8 次验证没有至少 0.2% 的相对改善，且已完成至少 3 轮、实际降过至少两次学习率，才停止。每次严格更低的验证 MSE 均保存 best；用于 patience 的显著改善参照独立维护，避免丢弃微小但真实的最佳权重。
- **保存/恢复**：best 与 latest 分开保存；latest 包含模型、优化器、调度器、早停、已见成员计数、采样位置、RNG 和累计预算。终末评价使用 best 自身的见过样本记录，不能用 latest 的样本暴露冒充 best。
- **时间**：用户明确授权本次助手直接执行，最长 **1 小时**，计入数据准备、验证、保存和终末评价；预留终末评价时间。早停不代表全局收敛，时间上限停止更不能称作收敛。不自动扩大预算或续跑。
- **终末诊断**：所选 best 在完整 2,500 dev 和 20,000 train 上计算同口径误差，若预算不足明确报告未完成；不打开封存测试。合法化 Q 使用固定 dev_quality100，独立记录预算。

已通过 [五例 train/dev 接口和降学习率检查](runs/20261016_workset_v2_preflight_seed0/results.json)。旧 E1/E2/E3 训练入口默认拒绝启动；只有明确历史复现才可加 `--legacy-protocol`，新研究训练不再默认使用旧成员或缺少验证控制的循环。

在工作区根目录可复现本次命令；正常训练可能在数分钟至 1 小时内因验证早停/资源保护停止。图特征缓存上限为单集合 6 GiB，预期 CPU RAM 数 GB、GPU 数百 MiB 以内（实际记录为准），输出仅保留 best/latest 与必要小型证据，建议预留磁盘 100 MB。

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.graph_proposal.train_supervised --run-dir experiments/graph_proposal/runs/20261016_deterministic_workset_v2_seed0 --budget-seconds 3600 --device cuda --threads 1 --batch-size 16 --validate-every 250 --early-patience 8 --min-epochs 3 --seed 0
```

Ctrl+C 等当前 batch 更新完成后保存。显式恢复使用原命令追加 `--resume`；源码、工作集、设备、batch、控制参数与累计预算必须匹配，不能用恢复重置一小时。训练前会将选定 train/dev 图特征缓存在本进程内，避免每次监测重复读取 500 个分片；这些缓存不写进数据集或 Git。

## 历史短预算阶段

状态：**短预算训练与 dev5 比较完成。确定性模型保留为 E1 学习参照；flow 暂停扩大投入，E4 不启动。** 用户已停止确定性训练；随后授权助手执行 flow 5 分钟训练及比较，未续训确定性模型。

最新证据见 [比较与研究判断](runs/20261016_short_budget_comparison_seed0/analysis.md)、[汇总数据](runs/20261016_short_budget_comparison_seed0/comparison.json) 和 [质量/训练曲线](runs/20261016_short_budget_comparison_seed0/comparison.png)。四组最终均 5/5 硬合法、无兜底；加权 Q：启发式 3.091515、确定性 **2.894369**、flow 单候选 3.645776、flow 四候选 3.441789。确定性相对同后处理启发式改善 6.38%、3/5 例同向，但 bbox gap 变差，仍未超过历史 baseline search 1.909019。仅为五例短预算信号，无收敛、泛化或正式比赛成绩结论。

确定性模型当前启动训练至暂停约 346.9 秒（累计含前次失败 374.9 秒）、2248 步；flow 298.1 秒、2557 步。实际训练时间和处理样本数不相等，不能把二者差异完全归因于生成式/确定性架构。两个最终 checkpoint 分别在 `runs/20261016_deterministic_train_seed0_r2/checkpoints/latest.pt` 与 `runs/20261016_flow_5min_train_seed0/checkpoints/latest.pt`；不复制或覆盖。

假设：图条件全局提议可以降低修复成本，且生成式多候选可能优于确定性回归。若同一后处理抹平提议差异或只靠兜底，停止扩大模型。固定 seed=0、不调宽度/层数/损失权重。主比较为冻结 dev5、15 秒/例上限、CPU 1 线程推理，失败计入官方中性时间加权质量分 Q；实际耗时另列。

## 实现和边界

- [data.py](data.py)：17 维合法输入特征，连接图、MIB/grouping 关系；仅 train 加载器接触监督坐标。拒绝普通块未遮罩的参考规格。
- [model.py](model.py)：共享 3 层、宽度 64 图编码器；确定性中心/对数宽高比回归与条件线性 flow matching。flow 固定 8 步 Euler，1 或最多 4 候选。自由块精确保面积，fixed 方向/尺寸、preplaced 坐标/尺寸锁定。
- [pipeline.py](pipeline.py)：原始提议 → 12 轮连续重叠修正 → 基于接触事件的最终构造。连续修正只动可移动坐标，不能保证合法；最终构造使用提议排序、形状和中心吸引。统一使用输入面积、连接、软约束代理选候选，不读取参考 Q。最终检查失败时单行合法兜底。
- [packing.py](packing.py)、[feasible.py](feasible.py) 从现有 baseline 当前源码复制于 2026-10-16，随后在本实验独立维护；不跨实验导入算法。新增提议中心吸引项；没有把此后处理称为 baseline 的原 search。
- [train.py](train.py)：用户运行的预算训练，可中断恢复；[run.py](run.py) 子进程只收到七参数合法输入，父进程独立评分三阶段。

模型输出对数宽高比截断至 [-5,5]，属于实验表示限制，不是比赛规则。软约束只提供特征/后处理偏好，训练未增加规则损失；当前 flow 最终收益不足，本轮不开展 E4 新增训练。

## 冻结训练成员

[train_members.json](train_members.json) 从冻结 train 分片按 `SHA256('E1_train_v1:' + shard)` 排序取 178 个分片，每片 0..111 行，共 **19,936** 个布局。生成仅读取索引，不读 train/dev/local_test 张量；实际载入按分片缓存，所有实验引用同一成员集合。成员哈希见文件。未改变原 80/10/10 划分。

```powershell
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -B -m experiments.graph_proposal.train --prepare-members-only
```

## 已有检查与判断

唯一完整几何、逐阶段评分、位移、耗时和源码快照见 [启发式 dev5](runs/20261016_heuristic_dev5_seed0/results.json) 与其 [配置](runs/20261016_heuristic_dev5_seed0/config.json)。启发式 5/5 硬合法、无兜底，加权 Q=**3.091515**，总求解 1.943 秒（平均 0.389 秒）。连续修正后五例仍有重叠，最终合法化是必要步骤。该入口允许最多 15 秒，但没有填满预算；历史 baseline search Q=1.909019 仅为历史参照，当前结果未显示质量优势。

[针对性反例](runs/20261016_heuristic_dev5_seed0/counterexamples.json) 验证非正方形 fixed 不旋转、拥挤 preplaced 在连续/最终阶段保持锚点，以及两个合法模式取坐标平均后重叠。通过反例只说明实现边界成立。

早先两个 [确定性一批](runs/20261016_deterministic_train_smoke_seed0/results.json) / [flow 一批](runs/20261016_flow_train_smoke_seed0/results.json) 训练检查共用同一个 train 布局 `worker_94/layouts_224.th:20`，CPU、batch=1。它们只验证载入、梯度、checkpoint 与模型提议接口，**不能作为学习质量结果，不能从这些 smoke checkpoint 续成正式对照**（正式 batch/device 不同）。后续的短预算学习质量证据见本页开头所链接的比较分析。

[CUDA 合成预检](runs/20261016_flow_train_smoke_seed0/gpu_preflight.json) 在 RTX 5080 Laptop GPU 上为两个分支各做一次三节点合成 backward/优化器更新，均通过，未读真实数据；峰值已分配显存约 65.6 MiB。这验证 CUDA 与确定性配置的可执行性，不代表实际 batch=4 的峰值显存或长训稳定性。相同目录的 `gpu_preflight.py` 保留复现入口。训练预算/源码身份检查在 smoke 后加固，旧 smoke 的恢复会拒绝源码不匹配；其实际源码快照保持原件。

## 原计划的一小时训练命令（本轮不再执行）

以下保留最初可复现入口及预算背景，已被用户的本轮短预算决定替代，不自动执行。最初两项各最多 1 小时，合计最多 2 小时，GPU、1 个 CPU 工作线程；预留显存 2–4 GB、磁盘 100 MB。实际短训峰值已分配 CUDA 内存约 67.3 MiB，不代表整个进程或设备总显存。时间上限是资源停止，不代表收敛。

```powershell
python -B -m experiments.graph_proposal.train --kind deterministic --run-dir experiments/graph_proposal/runs/20261016_deterministic_train_seed0 --budget-seconds 3600 --batch-size 4 --seed 0 --device cuda --threads 1
python -B -m experiments.graph_proposal.train --kind flow --run-dir experiments/graph_proposal/runs/20261016_flow_train_seed0 --budget-seconds 3600 --batch-size 4 --seed 0 --device cuda --threads 1
```

Ctrl+C 请求当前 batch 后保存并停止；恢复使用**原命令加 `--resume`**。累计预算不因恢复重置；源码、成员、数据位置/大小/修改时间、batch、device 等严格匹配，不静默恢复不同配置。当前仅保留一个 latest checkpoint，不称作最优模型。资源用尽后不自动扩大预算。

### 2026-10-16 Windows 保存修复与本次重跑

用户首轮 deterministic 训练已完成 25 步，但在约 27.3 秒时替换 `budget_ledger.json` 遭遇 WinError 5；未到旧版首次 60 秒 checkpoint，故原运行没有权重可恢复。原目录 `runs/20261016_deterministic_train_seed0` 的日志、失败记录、临时文件和源码快照保持原件。具体哪个外部读者占用了文件无法从该异常确认。

现版本预算按批追加到 `budget_ledger.jsonl`，不再反复替换同一预算 JSON；配置与 checkpoint 使用唯一临时文件并对 PermissionError 最多重试 3 秒。初始化、首批、之后每 60 秒及正常停止均保存 checkpoint；错误记录失败不再遮盖原异常。成员、模型、Loss、种子和划分未改。

[针对性验证](runs/20261016_windows_io_repair/verification.json) 实际打开 Windows 禁止删除共享的文件句柄，复现 WinError 5，并在句柄释放后通过重试保存；100 次替换/追加、30 个合成 batch、1 个恢复 batch 通过，初始化/首批/最终 checkpoint 及累计预算均核验。复现入口为 [check_io.py](check_io.py)，没有读取真实 train/dev/test 张量，未启动完整训练。

本次重跑从相同 seed 重新初始化，使用新目录；`--previous-attempt` 自动核验旧配置并将旧记录耗时向上取整至 **28 秒**计入原总预算，不假称恢复了丢失的 25 步。请在工作区根目录执行：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.graph_proposal.train --kind deterministic --run-dir experiments/graph_proposal/runs/20261016_deterministic_train_seed0_r2 --previous-attempt experiments/graph_proposal/runs/20261016_deterministic_train_seed0 --budget-seconds 3600 --batch-size 4 --seed 0 --device cuda --threads 1
```

恢复该新运行的方法是使用相同新目录及其他参数，**去掉 `--previous-attempt`，加 `--resume`**；但当前用户已明确停止，本轮不再续训。评估 deterministic 模型使用 `_r2/checkpoints/latest.pt`，不使用无权重的失败目录。相同保存修复也用于后续已完成的 flow 5 分钟训练。

训练完成再执行以下推理；每组最多 dev5，训练和评估分别运行。新结果目录不得覆盖已有记录。三个控制共享相同后处理与 15 秒上限，flow4 的额外候选/修正开销均包含在求解时间。

```powershell
python -B experiments/graph_proposal/run.py --strategies heuristic deterministic --checkpoint experiments/graph_proposal/runs/20261016_deterministic_train_seed0/checkpoints/latest.pt --seconds 15 --seed 0 --output experiments/graph_proposal/runs/20261016_deterministic_dev5_seed0
python -B experiments/graph_proposal/run.py --strategies flow1 flow4 --checkpoint experiments/graph_proposal/runs/20261016_flow_train_seed0/checkpoints/latest.pt --seconds 15 --seed 0 --output experiments/graph_proposal/runs/20261016_flow_dev5_seed0
```

只有学习控制相对同版启发式改善最终 Q 或明确减少修复成本，且没有高兜底率，才考虑扩大开发面板。不得从 dev 参考 Q 挑 flow 候选。当前确定性有有限信号，flow 最终收益未证实，不启动 E4 新增训练；local_test 始终封存。`summarize.py` 仅读取既有结果与训练日志生成当前汇总，不读数据张量。
