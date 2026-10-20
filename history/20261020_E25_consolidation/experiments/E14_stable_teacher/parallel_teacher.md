# 并行教师生成

此为旧版Q≤1.04筛选库说明。当前已改用 [25,000例全布局保存入口](archive_teacher.md)，请运行`generate_archive.ps1`。

当前入口：`generate_parallel.ps1`。网络权重、CPU后处理策略、MIB面积修正规则、Q≤1.04和各层配额保持原定义。执行方式为单GPU float32草稿、4个持久CPU后处理进程、主进程独立评分/编码/单点写库。GPU按最多8例批量推理，最多8个后处理任务在途，不预生成整个候选池。GPU解码后再次覆盖固定/预放置规格。

原串行库 `runs/teacher_q104_stratified_v1/` 保留；新库是 `runs/teacher_q104_parallel_v1/`。首次运行会持有原库锁，复制SQLite状态和所有已保留标签，核验哈希并保存源码快照。已处理的合格、淘汰和不兼容成员全部继承，不重新处理。迁移后请只运行新入口：旧库后续增长不会自动同步到新库。源码指纹没有绕过；新版本独立登记并保留原库来源。

```powershell
.\experiments\E14_stable_teacher\generate_parallel.ps1
```

在仓库根目录运行，不需另行激活环境。固定使用本地独立环境。默认累计48小时、60000次新求解预算，配额全部满足后提前结束；迁移前串行新增的5次检查和累计用时也继承。用户可显式扩展累计预算：

```powershell
.\experiments\E14_stable_teacher\generate_parallel.ps1 -MaxSeconds 345600 -MaxSolves 120000
```

停止（新库的STOP，不是旧库）：

```powershell
New-Item -ItemType File -Path .\experiments\E14_stable_teacher\runs\teacher_q104_parallel_v1\STOP -Force
```

停止派发新任务并完成已有在途任务后退出。继续时删除STOP并重跑原命令：

```powershell
Remove-Item -LiteralPath .\experiments\E14_stable_teacher\runs\teacher_q104_parallel_v1\STOP
.\experiments\E14_stable_teacher\generate_parallel.ps1
```

Ctrl+C也可中断；未提交任务保持pending，恢复时可能重算。派发前即计入求解预算，故中断不会重置尝试计数；该计数可能包含已派发但未落盘的任务，不等于已提交结果数。默认执行参数workers=4、draft_device=cuda、batch_size=8参与恢复身份校验；不要中途更改。

主进程持有独占锁，按派发顺序提交结果，不按最快完成者挑样。候选分层配额预留在途名额，防止同层超额；Q>1.04（包括Q>2）正常淘汰并继续。真实求解/硬检查错误停止，见last_error.json；调查后可重跑，未提交成员不会被悄悄跳过。STOP不取消已派发任务，Ctrl+C或错误时取消尚未开始的任务，仍在计算的任务等待结束。

数据读取复用datasets读取器，按分片LRU缓存最多8片；CPU进程只接收公开solver_inputs和草稿，不接收自由块参考或评分分母。主进程统一使用原scoring评分，验证/测试成员不进入新库。源分区及候选顺序继承原规范。

进度：新库progress.json；训练用清单：accepted.json；断点权威：bank.sqlite3；每次启动参数：sessions.jsonl；本轮源码快照：source.zip。运行中每25例提交导出进度，停止后再次导出。accepted.json只含实际通过门槛的标签；completed才代表所有分层配额满足。保留数据库和labels，不能只留导出清单。

线程配置：CPU worker的Torch、BLAS线程限制为1；Clarabel沿用原算法中的max_threads=2，主进程Torch为2。不以增加单例线程数替代跨例多进程。GPU约16GB显存，小批量和小模型不预期占满；实际显存、速度随任务和系统负载变化。新旧草稿后端存在浮点差异，墙钟限时后处理在并行下也可能改变完成步数，不能保证逐例结果完全相同；最终硬合法及Q门槛仍逐例强制检查。

本轮只运行最多5例功能检查，不启动长生成，也不据此宣称达到固定并行倍率。优先观察每小时合格标签数、各层接受率和尾部耗时；并行效率不等于模型效果提升。

已完成5例有效检查：全部硬合法，4例接纳、1例Q=1.259571淘汰后继续；首批4例由4个不同worker进程处理。STOP存在时零派发，删除STOP后恢复并新增一例，均通过。2023例原合格标签和全部旧已处理记录保持原样，新库当前2027例；所有保留标签校验值、任务版本及train分区成员已核验。证据：`runs/20261019_parallel_generation_smoke5/summary.json`。

准备过程中修复过可选依赖缺失及worker输入的官方评分API类型问题，失败未写入任何新标签，记录与修复前源码快照保留于新库。6次失败派发仍计入预算，因此当前计数16次＝迁移前5次＋失败派发6次＋有效并行检查5次。无需用户安装额外包。
