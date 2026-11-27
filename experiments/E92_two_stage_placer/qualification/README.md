# E92资格评价入口

2026-11-01准备。此目录不被训练导入，不改变训练源码身份；评价时另将本目录、共享评分／绘图和训练来源一起绑定源码快照。当前只准备代码及静态／保存证据检查，没有运行新模型推理、终末Q或图库渲染。

## 冻结与口径

- 入口先检查训练已正常早停或资源停止，再持有训练锁；训练仍活跃时拒绝运行。只允许F own-Loss best，核对历史最小F监测、best轮次、源码／数据及checkpoint身份。若2h结束仍无F checkpoint，直接记该项门槛失败，不强行切阶段或按Q改选P。
- 核对`runs/qualification_registration.json`在study.json登记的SHA256，以及E90原结果哈希；阈值不变。只有全部门槛通过后，deployment.json才允许成为修正训练的数据源。E42/E50不自动升级。
- `runtime/`是从冻结E90逐字复制的flash、模型和输入推理实现；已逐文件匹配E90终末配置中的原源码哈希，来源见provenance.json。没有跨实验导入私有算法；逐例计分与汇总沿用E90表达式，只有本实验内导入路径调整。
- 完整dev2000按原N截断padding后计算硬合法、交面积、覆盖并集、最差模块、外框和精确空白。F Loss复用训练函数，dev500另按生产监测路径复核；监测总量与所选best登记值应在1e-5内一致。
- 唯一一次flash500为4进程×2线程，参数与E90完全相同；失败例计入。断点只能补未完成成员，不重复已完成终末。E90已有结果只读复用，另报同worker配对区间；研究性容差通过不等于统计等价或未见测试成绩。
- 固定100例reference／E92／E90草稿图，种子20261025，只用保存坐标与公共visualization，统一ID／坐标尺度，无评分控件、不启动浏览器，不按结果筛图。
- 准备与评价共用本任务2h辅助资源上限：包括E92所有已计预检、task_auxiliary_budget.json及后续登记的修正器准备账本；本轮布局2h和条件修正训练2h分别计入，合计新任务不超过6h，历史成本保留。未达E90门槛即终止，不启动修正模型。

## 执行与恢复

仅查准备状态，不推理或评价：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E92_two_stage_placer.qualification.run --check-ready
```

训练正常结束并冻结唯一F best后，按已获授权执行：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E92_two_stage_placer.qualification.run
```

输出归本轮run下`qualification/`。可用`--stage raw|terminal|render|assess`恢复相应未完成部分；冻结身份不变，已有完整结果核对后直接复用。STOP放输出目录，或用SIGINT；停止保存已完成记录，耗时仍计入。异常先保留证据并诊断，源码修改不能绕过冻结身份；必要的报告修复应另留显式迁移及旧快照，不重复Q求解。封存测试从未开放。
