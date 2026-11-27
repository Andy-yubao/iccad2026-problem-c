# E27：单独提高重叠穿透惩罚

状态：固定500例验证已完成，500/500硬合法、零错误；加权Q1.061580，未优于同版E17的1.060518。草稿重叠程度降低，但不替换E25当前最佳。训练由用户停止于75轮，冻结best69（监测代理0.556924538），未恢复训练。核查见[runs/20261020_overlap1_v5_seed0/analysis.md](runs/20261020_overlap1_v5_seed0/analysis.md)及同目录completion_audit.json。独立复制E23训练/评价支架与E17/E25的小模型，保留E25当前最佳、E23容量对照和E26记录，不改旧源码或权重。源文件来源与冻结缓存身份见[provenance.json](provenance.json)。

## 问题与单因素设计

E23将参数从497,299扩大至1,370,771，改善了训练拟合，却未改善同版最终Q；开发集穿透平方和拥挤项也没有改善。因此本轮回到497,299参数（宽96、4层attention），与E17保持架构相同，仅提高训练重叠系数。

保留局部中心、形状、参考相对位置、参考接触、边界、MIB和bbox监督；监督不衰减。重叠仍是按局部模块尺度归一化的最小分离穿透深度平方，对可变模块对求和后按模块数归一化。**唯一目标变化是该项系数0.5→1.0**；不改为三次方，不加入拥挤/空白Loss，不生成教师数据，不修改MIB输入规则。

训练目标为：

`center_local + 0.25*shape + 0.25*relative_local + ramp*(1.0*overlap + 0.5*contact + 0.25*boundary + 0.1*mib + 0.1*bbox)`

`ramp=clip(epoch/2,0,1)`，沿用E17前两轮渐入。AdamW、学习率、权重衰减、梯度裁剪、seed0初始化及逐epoch样本排列规则均沿用原设置。从头训练，不加载已有best作微调，避免引入初始化差异。

## 固定监测与验收

训练Loss与选模指标分开：`total`是系数1.0的新训练目标；`geometry`仍以旧0.5系数计算，`repair=旧geometry+occupancy`继续用于best、LR和早停。`losses.py`的WEIGHTS保持旧值，实际训练增量在`loss.py`显式添加，不能据WEIGHTS里0.5误认为训练没有调整。

v5 train16,000/dev2,000、固定train/dev_monitor各500，沿用E25中唯一的E17原参考张量缓存。每轮在固定面板监测，训练和轮内验证不运行后处理。已有overlap、overlap_area、crowding、中心/相对位置误差继续记录；额外复用独立硬检查已得到的重叠对，记录冲突对数、最大冲突度及最大连通冲突簇的均值/P95。新诊断不进入Loss或选模，不增加三重覆盖的逐轮CPU扫描。

模型冻结后先核查完整dev2000，再对唯一best执行一次固定dev_monitor500的原E17后处理，2进程×2库线程。与E17/E23保持同版后处理，先隔离Loss作用；不直接与E25 steady的时间/Q混比。评价源码及backend独立复制，哈希登记。逐例输出保留，失败计入主统计，模型选择不使用后处理Q。

假设若成立，应同时观察到开发集穿透程度/高冲突负载减少、中心几何未明显退化，并最终减少救援或改善尾部/加权Q。若只减少重叠而中心漂移、bbox/线长或最终Q恶化，应否定直接替换；不能凭训练Loss更低称成功。E26好坏20例只作解释，不能作为新选模面板。开发面板已用于适应性研究，测试继续封存，Q仍为MIB派生任务的中性时间诊断。

## 预算、命令、日志

最多4小时或200轮，训练batch32；学习率固定`1e-3 → 2.5e-4 → 1e-4 → 5e-5`。第1～14轮保护，第15轮起连续3轮无明显改善下降一档，连续3轮无明显改善且已实际下降3次、最低档完成至少二轮训练与验证后早停。显著改善门槛仍为相对0.1%与绝对1e-5的较大者。best/latest、优化器、调度器、游标、随机状态和累计预算均保存。

在工作区根目录运行一个命令，依次核对缓存、从头训练/兼容断点恢复、冻结best并验证500例：

```powershell
& .\experiments\E27_overlap_weight\launch.ps1 -Resume
```

- 环境：`.conda/envs/iccad2026`；训练CUDA、CPU库线程4，终末评价总4线程。
- 控制台同步写入`runs/training_console.log`，重复执行追加日志。
- 正式模型/曲线：`runs/20261020_overlap1_v5_seed0/`，终末结果为其`final_evaluation/results.json`。
- 阶段状态：`runs/20261020_pipeline/pipeline.json`。
- 预计训练约30～90分钟，最多4小时；这是参考E17/E23的估计，新目标未做全程测量。终末500例预计另需约8～15分钟，受困难救援和系统负载影响。
- Ctrl+C或在正式运行目录放置`STOP`可请求安全停止；终末评价可在其`final_evaluation/`放STOP停止派发，在途最多2例完成后保存。恢复前移除对应STOP，重用上述`-Resume`命令。恢复保持原身份和累计预算，完成评价直接复用，不重选候选或重跑全500。
- 新目录无正式断点时从头训练；绝不把`runs/20261020_smoke5_seed0/`的实现检查断点用作正式续训。

实现检查限3例train与2例dev：验证batch32有限梯度、新Loss相对原Loss严格只多0.5*ramp*overlap、选模代理不变、模型/解码与E17逐字一致、图冲突诊断能区分同样3条边的星形/独立重叠、四档LR与最低档保护、两步停止/恢复、两例硬合法、单候选评价守卫及测试拒绝。最终结果见[runs/20261020_smoke5_seed0_r2/results.json](runs/20261020_smoke5_seed0_r2/results.json)。首次检查后将未修改算法源恢复为原始字节换行格式，随后在同五例上复核，保证未改动的控制算法哈希完全一致。只做实现检查，不启动正式训练或500例评价。

## 2026-10-21直接验证与停止规则迁移

实际训练仍使用旧早停8轮/最低档1轮实现，由用户在75轮后中途停止，原results和best/latest保持原样，不冒称新规则自动早停。最新记录为连续6轮停滞、三次下降、第三次下降在43轮；符合用户新设的3轮停滞与最低档至少2完整轮条件。训练源码已同步新规则，调度器纯状态检查通过；旧实现检查仍保留原源码快照，不改写其运行事实。

本次仅以`--finalize-stopped`显式授权冻结best69，原训练finished=false及user_stop状态不修改；最终评价登记在正式运行下的`verification_finalization.json`。原始完整dev2000尚未计算，本次按用户要求直接验证500例，不把它标为已完成。

后台日志为`runs/20261021_validation_background/stdout.log`和`stderr.log`，启动身份为同目录`background_launch.json`。进度为正式运行`final_evaluation/progress.json`；完成结果为`final_evaluation/results.json`。同一个冻结best的中断验证可用下列命令恢复，不重新训练、不更换候选：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E27_overlap_weight.final_evaluate --run .\experiments\E27_overlap_weight\runs\20261020_overlap1_v5_seed0 --device cpu --workers 2 --resume --finalize-stopped
```

本次结束后不自动恢复或重训；用户新规则改变了源码身份，原训练若要续跑须另行明确调度/身份迁移，不能直接绕过恢复检查。测试继续封存。

## E27模型＋steady快速后处理：单线程验证

独立入口`fast_eval/`复制E25快速求解器，来源哈希见该目录provenance.json；不跨实验导入私有算法。只绑定E27 best69并将计算限制为1个worker×1个库线程：Torch intra/inter-op、BLAS/OpenMP、Numba、HiGHS和Clarabel均为1。steady的时间预算和策略不改，因此时间预算内的迭代次数及最终Q可能受线程变化影响。不能把与历史四线程E25的差异全部归因于模型。

2026-10-21轻量检查仅执行同面板第1例，1/1硬合法、零错误，结果保存在runs/20261021_fast_single_smoke_steady_monitor1/results.json；没有启动500例。本入口在每例计时内重新构造公开特征、推理、解码、后处理，模型/文件加载和首次原生初始化另列；缓存仅保留公开case，不运行500例预推理。

在工作区根目录执行：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E27_overlap_weight.fast_eval.run --tag 20261021_e27_single --configs steady --count 500 --direct 2>&1 | Tee-Object -FilePath .\experiments\E27_overlap_weight\runs\20261021_e27_single.log -Append
```

日志：runs/20261021_e27_single.log；结果：runs/20261021_e27_single_steady_monitor500/results.json；逐例中间结果：同目录partial.json。重复同一命令复用已完成逐例结果；在该结果目录放置STOP将停止派发，移除后可续跑。原始旧后处理500例和E25保持原样，测试继续封存。
