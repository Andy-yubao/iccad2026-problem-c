# E9 当前执行方案：先争取平均 Q＜1.20

2026-10-18，用户在了解固定 dev100 的 MIB 冲突及下界后，将本阶段目标调整为平均 Q＜1.20；后续另选不存在这类冲突的测试数据。本文件是当前训练规格，初始＜1.10计划保留为研究历史。

## 当前结论：中心修正的收益未得到额外面板支持

中心修正训练已由用户执行完成：10轮验证早停，best为第9轮step5625，累计17.91分钟，完整dev2500原始预测误差评价完成。dev100从1.211556降到1.210118，均值仅改善约0.12%；随后冻结同一候选，在已固定的confirm100完成一次评价，均值从1.232217升到1.239278，尾部也变差。两组输出均100/100硬合法；本次确认由助手在900秒保护预算内执行，实际287.4秒，无测试读取。

[完整复核与证据](runs/20261018_center_refiner_confirm100_seed0/analysis.md)。目标＜1.20仍未完成。继续保留旧模型＋E8作主参照，中心修正候选只保留研究证据，不提升默认配置、不追加这套方案的训练。下一轮需要新的可证伪机制，先核查可避免MIB罚项及真实后处理反馈，不以放宽早停替代机制改进。

以下是已完成阶段的原设计与历史命令，仅用于复现，不能读成待启动或待续跑任务。

## 已完成阶段：冻结旧模型，只学中心修正

首轮两组正常完成，总计34.5分钟，均10轮早停，未达到＜1.20。继续原geometry训练的第5轮Q=1.214180；新目标代理最优Q=1.215390；两组均保留旧模型Q=1.211556。[完整候选与分析](runs/20261018_quality_alignment_seed0/analysis.md)。不续跑这一轮、不因代理指标小幅下降延长早停。

接着在同一固定5例做了两类参考中心oracle诊断，均保持原形状和预放置位置、冻结E8及预算：[整体平移](runs/20261018_group_translation_probe5_seed0/analysis.md)中，按组整体平移均值从1.151565变为1.157157，没有收益；[逐模块中心纠正](runs/20261018_center_blend_probe5_seed0/analysis.md)中，纠正25%/50%位置误差后均值为1.144817/1.135289，后者5例全部改善。参考仅用于诊断，不是可部署输入；五例不构成正式模型选择或泛化结论。

因此下一轮选`center_refiner`：旧attention模型及其形状预测完全冻结，仅训练约1.2万参数的小位置修正模块，输入只含公开特征、旧模型特征/几何和连接/分组上下文。它逐模块输出中心偏移，不增加关系头；不强制组内所有模块同向平移，也不采用已停止的rectangle架构。新模块末层零初始化，初始输出与旧模型相同。训练仍用原geometry目标，避免同时恢复已失败的新损失；官方评价、E8及面板保持不变。

对照是冻结旧模型和首轮已完成的继续训练控制组。收益若出现，只能归因为“冻结骨干、只学中心修正”的完整方案；不能拆称为单独冻结或单独新增容量的贡献。末端选中旧权重则如实记为未改善。

本轮已执行命令（历史复现；已完成目录不可覆盖，也不建议追加续训）：

```powershell
Set-Location E:\project\ICCAD\2026\problemC
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.train --run-dir experiments/E9_model_postprocess_alignment/runs/20261018_center_refiner_seed0 --variants center_refiner --max-epochs 30 --arm-seconds 3600 --budget-seconds 5400
```

粗估20–60分钟，单组累计上限1小时，整次累计上限1.5小时，均包含相应训练/监测/质量评价；超过预算按资源停止记录。仍建议预留4GB显存、12GB主存、1GB磁盘。工作集20,000、固定监测500、batch32、初始LR1e-4、下降/早停及第5/15/30轮和代理best的完整dev100评价规则沿用下文。best/latest、selection.json、analysis.md/comparison.json及停止/恢复方式也相同；恢复只能在原命令末尾加`--resume`，源码/配置/预算须一致。

实施前的[人工样本功能核验](runs/20261018_center_refiner_preflight_seed0/results.json)已通过：初始草稿与旧模型逐值一致，训练后骨干权重及形状输出保持不变、中心确实更新，恢复训练与不中断训练权重逐值一致；可训练参数12,194，该功能核验没有真实样本训练。随后的正式训练与确认结论见本文件开头，未达到目标。下文继续保留首轮设计、命令及通用执行说明作为历史参照。

## 1. 这一轮具体改变什么

先固定 E8 `conic_robust`，保留原来的普通例1.3秒、整例5秒目标和救援机制。第一轮不增加模型层数，也不恢复停止的 occupancy 损失或 rectangle 架构。

两组从同一个 E7 `attention_geometry` best 的模型权重开始，优化器和训练计数在 E9 重新建立；不是恢复 E7 的旧训练任务，不修改其权重和 STOP 标记。

| 组别 | 训练方法 | 用途 |
|---|---|---|
| `geometry_control` | 用原 geometry 目标继续训练 | 测量追加训练本身的贡献 |
| `quality_aligned` | 保留原位置、形状、接触、重叠等监督，调整质量目标 | 检验与最终质量对齐是否更有效 |

新目标有三个具体变化：不再惩罚比参考更小的 bbox，只惩罚面积超额；增加加权曼哈顿线长超额；加强公开边界要求。线长计算保留完整的块间边、引脚边和权重，不能用归一化 attention 邻接或引脚平均位置替代。

计算式为原 geometry，减去原0.1×对称 bbox 损失，再加0.1×面积超额、0.25×线长超额、0.5×原边界损失。新增差异在前2个 epoch 逐步引入。训练的参考归一化采用 train 参考布局直接计算的值，是训练代理，不替代官方 Q，也不假定参考 Q=1。不可兼容 MIB 不增加额外惩罚，沿用原 geometry 中的兼容对处理。

预期信号来自此前五例同实例诊断：改善中心组织比仅替换形状更有帮助。它只支持做首轮验证，不保证新目标有效。如果仅代理 loss 下降而最终 Q 不降，就停止扩展这组目标，再考虑全局组织架构；不自动进入多轮调权重。

## 2. 固定的数据、监测与选择

- 训练使用原 workset_v2 的20,000例；固定 train_monitor500 和 dev_monitor500，每个完整 epoch 监测，初始 step0 也记录。无测试读取。
- 两组同初始化、seed0、batch32、AdamW，初始学习率1e-4、weight decay1e-4，梯度范数上限1；attention宽96、4层，仅输出中心XY与log宽高比。
- 两组统一按 dev_monitor500 的 `aligned` 代理执行学习率下降与早停，分别报告原 geometry、中心误差、形状误差及新增各项。不能直接把代理最优写成 Q 最优。
- 学习率停滞 patience2、相对改善阈值0.2%（绝对阈值0）、下降倍率0.5、cooldown1、最低3e-6。早停需至少10轮、连续10次未显著改善，并距最近降学习率至少3次验证。最多30轮是资源上限，不是收敛证明。
- 第5、15、30轮及终末代理 best 是预先固定的 dev100 质量候选；相同模型参数不重复评价。质量节点保存可恢复权重，最终按完整100例真实 Q 选择。评估失败计Q=10，不挑子集；未完成面板不参与选择。
- 既有 E7＋E8 的完整 dev100 作为冻结参照，均值1.211556，始终可以保留。若新模型不优于它，selection.json继续指向原权重，不能称为新模型改善。
- `best.pt` 是监测代理最优，`latest.pt` 支持完整恢复；`selection.json` 才是实际完整 dev100 选中的模型及评价路径。终末对所选模型做完整 dev2500 原始预测误差核验，资源不足时明确留为 pending。

每组质量选择最多4个固定候选；开发选择次数、随机种子和所有失败均披露。主目标仍是固定 dev100 算术平均 Q＜1.20，同时报告官方块数加权 Q、P95/最大值/尾部、硬合法及实测耗时。最终额外的 confirm100 单列复核，不用它反复选择 checkpoint，也不因它被称为确认面板就声称是独立未见测试。

## 3. 正式训练命令与预算

**以下是待用户执行的正式命令，助手没有启动该训练。** 按既有速度粗估约1–3小时，整次累计资源上限4小时，每组累计上限90分钟，均包含该范围内的监测/质量评价；准备开销计入整次预算。实际速度受硬件和案例影响，不能把预算用满称为收敛。预计预留约4GB显存、12GB主存、1GB磁盘；代码限制单集合特征缓存6GiB、合计8GiB，实际用量写入 data_audit/results。

```powershell
Set-Location E:\project\ICCAD\2026\problemC
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.train --run-dir experiments/E9_model_postprocess_alignment/runs/20261018_quality_alignment_seed0 --variants geometry_control quality_aligned --max-epochs 30 --arm-seconds 5400 --budget-seconds 14400
```

无需修改 base 环境。输出归上述新目录：根目录记录源码快照、实际命令、环境/数据位置/冻结哈希与总体预算；各组保存日志、checkpoint、质量节点及 selection；根 analysis.md/comparison.json 自动比较同面板结果，不重复保存参照坐标。终末模型不优于原权重时，表格会保留原权重选择事实。

停止：Ctrl+C会请求在安全边界停止并保存，也可在本次运行根目录创建 `STOP`。评价子进程忽略Ctrl+C，由父进程等待当前有界子问题结束再保存。正常停止可按原命令追加 `--resume` 恢复，数据、配置、源码及预算均须相同，累计时间不重置；先移除本次运行的STOP，不能改E7旧STOP。

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.train --run-dir experiments/E9_model_postprocess_alignment/runs/20261018_quality_alignment_seed0 --variants geometry_control quality_aligned --max-epochs 30 --arm-seconds 5400 --budget-seconds 14400 --resume
```

`arm_budget`、`epoch_cap`、`validation_early_stop` 的已结束组不会因恢复再次训练；用户中断和未耗尽总预算的中断组才继续。达到总预算时不自动加预算。源码改变则拒绝混跑，应恢复源码快照或另开新运行；强制结束进程可能丢失当前未完成步骤及最近一次保存后的计时，不能拿它规避预算。质量或dev2500未完成时，不在恢复说明中伪装成已完成。

## 4. 训练后评价与查看

根目录 `analysis.md` 对比原权重、继续原目标、新质量目标；`comparison.json`给出新目标减控制组的均值差，区分新目标贡献与追加训练收益。只重建报告：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.summarize --run experiments/E9_model_postprocess_alignment/runs/20261018_quality_alignment_seed0
```

训练结束后，对已经选定的模型另做 confirm100 评价。以下以新目标组为例；如果控制组更好，就改用其 selection。该命令不训练，通常约3–6分钟，累计保护30分钟；中断后重复同一命令会继续未完成案例，完整结果不重复求解，权重或源码变化则拒绝混跑。

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E9_model_postprocess_alignment.evaluate --selection experiments/E9_model_postprocess_alignment/runs/20261018_quality_alignment_seed0/quality_aligned/selection.json --panel confirm100 --output experiments/E9_model_postprocess_alignment/runs/20261018_quality_aligned_confirm100_seed0 --budget-seconds 1800
```

评价从公开规格与模型草稿进行求解，参考只在父进程事后计分。文件交换、模型推理、后处理分别计时；不把复用的旧指标当作重新在线运行。官方中性时间 Q 与实际耗时分开，不能据此计算官方比赛名次。

## 5. 已完成的功能核验

[训练入口核验](runs/20261018_quality_alignment_preflight_seed0/results.json)：5个人工构造案例，两组有限步数的梯度与断点恢复检查通过，中断恢复与不中断得到的模型参数逐值一致；精确加权线长与官方函数差异小于3e-6。另仅在1个此前已访问的开发案例无梯度前向，草稿与E8缓存逐值相同，冻结后处理Q也相同。

[评价入口核验](runs/20261018_quality_evaluation_gate1_seed0/results.json)：同1个开发案例检验预算停止、恢复完成、不重复求解和变更模型拒绝恢复；MIB兼容筛选的正例及矛盾反例也通过。人工检查产生的checkpoint明确标为fixture_only，正式评价入口拒绝把它们作为科研模型。上述核验没有正式训练或泛化结论，不把其数值当作新模型成绩。

## 6. 后续更换兼容测试面板

当前＜1.20研究仍使用既有开发面板。后续另立带版本的新测试面板，不改 split_v1/workset_v2 原件，不把原测试迁入训练，不按模型Q筛选。`compatibility.inspect_mib`已实现只读取合法公开输入的局部兼容规则：

1. 无锁定块的MIB组，其所有目标面积的1%合法区间须存在公共交集，可构造共同宽高。
2. 有fixed/preplaced的组，公开固定宽高须一致，而且共同尺寸的面积落入全部自由成员的合法面积区间；使用精确规格与小裕量，属于保守筛选。
3. 不修改或删除原任务的MIB约束来凑数；数量不足就报告不足。筛选前固定成员排序、数量、覆盖与失败处理，筛选率和输入分布变化必须披露。

这只能证明不存在所检查的“MIB同形与面积/固定尺寸”局部冲突，不能证明全部边界、分组与几何约束同时可行。以后若要求全约束兼容，需要额外的合法布局见证。由于筛选改变分布，新面板分数不能直接与旧面板相减作为模型收益；旧面板保留作历史对照。

本轮未扫描、筛选或评价封存测试。后续实际读取、成员和预算在用户明确开展该面板时另行登记；不能把本次未来安排解释成恢复任何已停止的旧测试。
