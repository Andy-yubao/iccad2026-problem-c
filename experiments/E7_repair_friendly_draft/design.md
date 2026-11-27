# E7 历史设计与实现规格（已终止）

> 2026-10-17：用户已明确终止本阶段，停止采用新增occupancy损失，不继续矩形架构或未完成组合。现行结论见[阶段报告](conclusion.md)。下文所有“当前”“恢复”“运行命令”均为历史规格，不能作为继续执行的授权；不移除STOP，不自动补训练、dev100或test50。源码与原运行记录保留供复核。

## 本轮任务

用户2026-10-17确认：复杂重叠和内部留白是值得检验的研究直觉，即便改善后尚不能立即加速当前修复器；取消5倍/10倍门槛，只要求耗时不增加。目标为50例在当前后处理下逐例Q算术均值＜1.2、进阶＜1.15。允许查看测试图像、诊断和成绩并迭代，不把受反馈影响后的测试成绩包装成独立盲测。用户明确要求本次助手不启动长训练，交付命令。

保留禁止关系头的既有边界。测试不参与梯度、训练目标或验证早停；train/dev/test原分片与工作集不变。本轮不改后处理，其后续调整另行留证。

## 对照与判据

| 运行名 | 架构 | 训练目标 | 能回答的问题 |
|---|---|---|---|
| attention_geometry | E6的96宽、4层attention | 原geometry | 同种子从头训练的控制 |
| attention_occupancy | 同上 | geometry＋occupancy | 单独增加占用目标是否改善 |
| rectangle_geometry | 96宽、4层矩形感知attention | 原geometry | 新架构是否改善 |
| rectangle_occupancy | 同上 | geometry＋occupancy | 两机制的组合与交互 |

四组seed0、完整train20000、batch32、AdamW、学习率1e-3、weight_decay1e-4、梯度范数裁剪1。相同架构的两组完全相同初始化和样本顺序，不使用E6权重续训。架构参数量与真实耗时分开记录，不伪称等参数。E6原best作为外部冻结控制加入每次后处理评价。**用户随后接受：首组使用原训练策略并人为停止，后三组使用更积极的fast策略，不能再称四组严格同训练策略/同收敛条件的因果消融；只在结果值得继续时考虑补同策略控制，不自动重训首组。**

晋级要同时检查：真实草稿多重覆盖、重叠簇和内部空白是否改善；最终硬合法率、Q及B/G/M；后处理与总耗时是否增加。几何改善但当前Q或时间未改善仍保留其机制证据，不能说已经达成终局任务。若变成单纯推散或过度挤压，且修复质量/成本受损，停止扩大该机制。5例仅排错，正式选择看dev500/dev100。

## 模型变化

E6 attention的几何偏置只包含中心差、绝对中心差和公开连线/MIB/group关系。新rectangle块每层先按面积与规格解码当前XYWH，形成11维内部块对特征：局部尺度归一化的中心差2、边间距2、真实交集宽高2、宽高比对数2，加公开关系3。用于attention偏置，并将交叠面积相对小块面积作为邻域权重，聚合8维几何特征和log拥挤度，形成节点修正。

初始三通道草稿仍由head生成；每层残差更新用tanh限制到块局部尺度，XY步长为sqrt(area)，log宽高比步长为0.5。残差head初始为零；端到端训练可改变全局初值和每层修正。fixed/preplaced通过公开规格解码锁定，只有内部几何特征，没有额外关系输出或辅助关系头。输出始终是每块中心XY与log宽高比。

此架构不保证草稿合法，也没有把昂贵LP嵌入前向。它增加的推理成本将实测计入总耗时。

## 新损失与可反驳点

保留E6 geometry的局部中心、形状、参考近邻、穿透、组接触、边界、MIB与参考bbox损失，权重不变。新目标只增加：

| 项 | 公式及目的 | 权重 |
|---|---|---:|
| overlap_area | 无序块对交集面积之和 / 总模块面积；不只看分离所需的较短穿透轴 | 1.0 |
| crowding | 每块与其他块交叠面积/自身面积之和，超过0.5部分平方，再逐块平均 | 0.25 |
| void | 在当前bbox内32×32网格上，1−软覆盖并集的平均 | 0.25 |
| compactness | max(log(bbox面积/总模块面积),0)的平方 | 0.15 |

void使用每块四侧sigmoid软覆盖；侧温度为块宽高的4%，最低归一化0.002。并集用1−各块未覆盖概率之积，bbox随当前几何可微变化。crowding是复杂冲突代理，不是精确三重覆盖积分；网格也不是精确空洞测量。不能以代理自身下降证明研究假设成立。不可移动preplaced之间的块对罚项排除，避免无可修自由度的常数项主导。

原有几何规则项及新occupancy项在前2轮线性引入，监督项始终生效。联合使用的原因是：只罚碰撞可能推散，只压bbox/留白可能挤成重叠。reference仍仅是train监督和dev/test外部诊断，不进入网络或求解器。

正式图表使用精确矩形扫描线积分：覆盖并集、二重/三重覆盖、最高覆盖重数、bbox内部空白；另算冲突度数、最大冲突连通分量和至少3块的冲突簇。图中坐标轴外侧空白不算布局内部空白。相同草稿再测实际修复位移/形变和优化轨迹，检验代理与修复成本之间的联系。

## 学习率与早停

每完整epoch结束验证固定dev_monitor500，并评估固定train_monitor500；初始step0也记录。四组统一使用完整权重的`repair = geometry + occupancy`作为best、降学习率和早停监测，不随训练warmup改变监测定义。它是草稿开发代理，不是最终Q；训练后须实际后处理评价。

当前后三组采用`--schedule fast`，原规格保留为`--schedule original`：

| 参数 | original（首组） | fast（后三组，当前默认） |
|---|---:|---:|
| 相对显著改善阈值 | 0.2% | 0.5% |
| LR patience | 4，即连续5次无显著改善 | 2，即连续3次无显著改善 |
| 下降倍率 / cooldown / 最低LR | 0.5 / 1 / 3e-6 | 相同 |
| 早停停滞次数 | 16 | 10 |
| 最低轮数 / 至少下降次数 | 10 / 2 | 相同 |
| 最后降LR后至少观察次数 | 5 | 3 |

早停四条件仍须同时满足，给下降后的训练留出观察机会。best保存任意更小监测值，显著阈值只用于LR和早停计数；小于0.5%的改善仍会保存best。默认max_epochs=0表示无轮数上限，无墙钟上限。可选保护轮数并不等于收敛。

首组已于epoch86.0544、step53784由用户停止；best为step48125/epoch77、dev_repair0.505574625。完整旧config/results/log/源码快照和唯一权重原地保留，不把user_stop改写成收敛。首组完整dev2500评价仍pending；后续三组不是从其latest续训。

best/latest均保存模型、优化器、scheduler、随机数、epoch、batch cursor、seen和早停计数，latest约每60秒以及每次监测、正常停止保存。源码、面板、数据划分与参数在恢复时严格核对。源文件哈希及源码zip与运行同存；不在这里建立另一个公共算法库。

训练结束自动对best做完整dev2500原始几何损失评价，保存dev2500.json；若该阶段被中断，恢复只补缺失评价，不自动重训已早停分支。train_monitor中的已见/未见数量分别记录，完整第一轮前不把整个面板当成已训练样本。

## 训练命令

以下命令仅记录历史复现规格。本阶段已终止，不作为继续执行计划；新授权前不要启动或恢复：

```powershell
Set-Location E:\project\ICCAD\2026\problemC
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -B -m experiments.repair_friendly_draft.train --run-dir experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0 --schedule fast --variants attention_occupancy rectangle_geometry rectangle_occupancy
```

该命令只按顺序训练后三组并做其终末dev评价，没有自动测试/后处理，也不恢复第一组。正式训练预计小时级，三组可能持续数小时；更积极的规则不能保证具体停止时间。默认GPU batch32；显存预计约1–5GiB，首组实测峰值约1.09GiB；CPU特征缓存预计数GiB，建议保留至少8GiB可用内存。三组checkpoint、日志与源码预计低于0.5GiB，不包括既有数据/环境。若batch32显存不足，应停止后用新目录从头启动，并记录差异，不静默更改恢复规格。

Ctrl+C由信号处理器转换成请求，在当前完整step/监测后保存latest。更稳妥的异步停止方法：

```powershell
New-Item -ItemType File -Path experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0/STOP
```

恢复前移除自己创建的STOP文件；删除范围仅该停止标志，然后沿用原命令加--resume：

```powershell
Remove-Item -LiteralPath experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0/STOP -ErrorAction SilentlyContinue
python -B -m experiments.repair_friendly_draft.train --run-dir experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0 --schedule fast --variants attention_occupancy rectangle_geometry rectangle_occupancy --resume
```

不要反复Ctrl+C强杀保存过程；若进程被外部终止，latest只承诺恢复至最近完成的step，不承诺保留在途计算。已早停的分支不会因--resume再次训练。新种子、新参数或从头重跑须另用run目录。

## 后处理、测试和图集命令

冻结后处理逐文件复制自E6；来源与SHA-256见inherited_sources.json。无MIP，120秒保护上限、24次形状迭代、重加权与严格输出守卫均保持原样，不缩时限换取虚假加速。每次评价包含原冻结E6模型，所有模型在同一批成员上实际推理和修复；同机2CPU线程，预热GPU，并逐例轮换模型顺序，减少首例和固定顺序的计时偏差。

训练完成后先执行dev100。五模型每例上限120秒，100例最坏约16.7小时，按既有平均10.854秒仅作粗估约1.5小时，真实值受新草稿影响；因此此推理也交用户运行，不由助手拆分启动。

```powershell
python -B -m experiments.repair_friendly_draft.evaluate --train-run experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0 --additional-train-run experiments/repair_friendly_draft/runs/20261017_factorial_seed0 --output experiments/repair_friendly_draft/runs/20261017_dev100_seed0 --panel dev100
python -B -m experiments.repair_friendly_draft.render --run-dir experiments/repair_friendly_draft/runs/20261017_dev100_seed0
```

test50命令显式访问用户授权的固定50例；不自动读取其余2450例。50个worker从40..99按预先哈希排序选取，各选原v2工作集内一个成员；整个0..39保守排除以避开旧957例和可能在途分片。不宣称按模块数精确分层。所有训练检查与预选面板过程均不读取test张量。

```powershell
python -B -m experiments.repair_friendly_draft.evaluate --train-run experiments/repair_friendly_draft/runs/20261017_remaining_fast_seed0 --additional-train-run experiments/repair_friendly_draft/runs/20261017_factorial_seed0 --output experiments/repair_friendly_draft/runs/20261017_test50_seed0 --panel test50
python -B -m experiments.repair_friendly_draft.render --run-dir experiments/repair_friendly_draft/runs/20261017_test50_seed0
```

可以用--variants rectangle_occupancy只评一个已完成训练的候选，仍自动加入冻结E6控制；这不会改变面板或评分。测试图像和诊断可反馈到后续设计，需记录哪次反馈影响了什么选择，后续结果披露复用。训练入口从不读取test。若需要独立泛化结论，应另立尚未反馈的终末评价，当前不擅自扩读。

联合评价优先读取新训练目录，缺失的attention_geometry从--additional-train-run旧目录按引用读取；不复制权重，不改旧结果。评价config逐一记录实际checkpoint路径/哈希及各源运行的scheduler和早停参数，明确首组与后三组差异。旧运行不要使用当前代码直接--resume：新源码会触发恢复校验拒绝，并且用户已经明确不继续训练首组。

评价每完整case写入唯一cases.jsonl，坐标不在汇总里重复保存；异常保留Q=10。支持Ctrl+C/STOP在当前case后停止；同一评价命令加--resume跳过已持久化case，源码/配置/权重不一致时拒绝拼接。results.json报告已完成数与预定分母，不把未完成当成完整成绩。图集只读保存坐标，不再读取数据或运行求解器。

报告Q算术均值、块数加权Q、尾部、合法率、HPWL/bbox gap与B/G/M；时间分别为特征准备、前向、完整后处理和端到端，均值/中位/P95。阶段计时使用高精度perf_counter，避免Windows的monotonic粗粒度将短前向记成0；CUDA计时前后同步。失败计入Q分母；计时仅在双方正常返回例上比较时另标数量，不用缺失失败时间声称更快。trace_subproblems是保存轨迹条目数，不能冒称内部所有LP调用次数。

## 实现验证范围

[五例检查入口](preflight.py)验证训练/单例推理特征一致、隐藏标签变动不影响输入、padding/逐例一致、模块置换等变、3通道输出、固定规格、有限梯度与一步恢复等价；合成几何检验精确面积/多重覆盖和损失梯度方向。它只在5个固定dev样本上做每组4个优化step，不是正式训练、收敛或质量证明。

另以--smoke在同一5个dev开发夹具上检查真实训练循环：最多2轮，训练/监测故意复用夹具，所有输出标为非科研训练。正式评价拒绝使用这类checkpoint；评价--smoke只检查第一个dev夹具及一个新模型，加冻结控制，仍不读取test。保存JSON与checkpoint的原子替换对Windows短暂文件占用作有限重试，不截断旧checkpoint。结果及运行轨迹见README的实现检查链接。

```powershell
python -B -m experiments.repair_friendly_draft.preflight --output experiments/repair_friendly_draft/runs/20261017_preflight_seed0
```

已有目录不能重跑覆盖；排错后用新后缀目录，保留必要失败证据及说明。阶段结束按项目规范整理，保护旧参照和唯一checkpoint。
