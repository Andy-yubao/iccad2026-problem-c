# E114 4+12：约500ms/例的大预算后处理对照

用户于2026-11-08要求增大4+12的后处理预算、运行500例，并明确复用原固定dev500、后台运行。仅本次评价，不训练、不读取封存test，不改变模型权重、输入、Loss或原后处理结果。

冻结E114 staged best48，checkpoint SHA256为bd6906668cc2584133bb863590825874a951e7c24d8085280b7fd821c13f6937。原独立500例结果完整保留在`runs/staged_seed0/evaluation_no_mib_dev500`，本次输出独立保存。

原完整平均耗时约176ms/例。本次单一预登记方案将原工作预算乘3：主shape上限由[4,4,7,8]变为[12,12,21,24]，扩展2→6、救援精修4→12；shape原生迭代100→300、projection80→240；certified polish接受次数4→12、候选240→720。种子、几何算法、目标、合法性与数值容差、现有提前停止条件全部保持。

500ms/例作为平均完整耗时目标，不强制每例截断，也不强制容易例用满时间。不按dev Q扫描预算，不剔除慢例或坏例；真实均值/P95/最大、超500ms/1秒例数均保留。若平均未接近500ms，照实报告，不自动追加预算扫描。

固定500例按原顺序正式评价，重跑同一冻结模型以测量当前完整时间；逐例核对原始draft，允许的最大数值偏差预先固定rtol/atol=1e-6，并另外报告逐位一致例数。完整时间包含公开输入复制、特征、传输、网络、全部修复/救援/回退和原规格独立硬检查；评分、载荷读取、模型加载、预热分列。Q保持原始面积/无MIB的本地研究分数，时间因子中性，不称官方比赛成绩。

- `stdout.log`：每10例和前5例打印进度、单例Q与完整耗时。
- `stderr.log`：错误和底层求解器警告。
- `launch.json`：后台进程、完成状态和退出码。
- `output/state.json`：实时阶段及完成例数。
- `output/cases/0000.json`等：全部逐例布局、trace、工作量、质量、独立硬检查和耗时。
- `output/results.json`：本次汇总。
- `output/comparison_report.md`、`output/comparison.json`：与原固定深度预算的成对比较，含所有退化和尾部坏例。

本目录STOP或output/STOP可终止评价；失败保存证据，不自动重试。原训练STOP和其他实验状态不改。新增求解代码从同实验evaluation_staged复制到本目录独立维护，来源哈希见source_manifest.json。

后台模块：`experiments.E114_supervision_placement.evaluation_budget500ms.launch`。父控制器保留持续后台会话，Windows子进程隐藏窗口；不创建定时任务或自动唤醒。
