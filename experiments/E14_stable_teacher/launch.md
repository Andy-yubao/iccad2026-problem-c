# E14 运行、停止与恢复

使用仓库独立环境，从仓库根目录执行。旧E12/E13入口不用于新数据。

## 数据规范

唯一新规范是[derived_workset_v3.json](../../protocol/derived_workset_v3.json)。已物化train16,000和dev2,000；其原始分片归属分别为split_v1的train/dev，没有交叉。local_test2,000是事前登记的目标和固定选择规则，保持封存，未读取或物化。本轮开发100与复核500均来自dev，后者每worker5例且与前者不交叉。

本机输入修改审计在[runs/data_v1](runs/data_v1/partition.json)：MIB组面积平均、固定和预放置规格不变；不相容任务拒绝。生成时必须配套读取新输入面积，禁止把新标签绑定到旧面积。

重新构建本机缓存（按规范中的确切成员，不重新筛选，不覆盖既有文件）：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E14_stable_teacher.rebuild --split dev
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E14_stable_teacher.rebuild --split train
```

## 冻结评价

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E14_stable_teacher.evaluate --panel development100 --output experiments/E14_stable_teacher/runs/<新开发运行>
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E14_stable_teacher.evaluate --panel confirmation500 --output experiments/E14_stable_teacher/runs/<新复核运行>
```

这两条是复现命令，不是自动再测授权。源代码、权重、成员与预算身份一致时可加`--resume`继续中断运行。`--baseline-run <既有同面板目录>`只复用保存的E10完整底解和计时，用于机制对照，不能改变底解成员。所有不可行和异常计入Q=10。

E10总后处理目标5秒、普通路径1.3秒；E14最多增加16秒有条件救援，目标不是硬实时保证。模型CPU推理、后处理与新增救援合计计时；模型加载、读盘、写盘及评价本身不计入求解时间。时间预算会影响实际迭代数，因此不同调用的原E10结果可能有微小差异；主比较共享同一底解，隔离新增机制的因果影响。

## 新版教师生成

仅在100＋500验收通过后运行：

```powershell
& .\experiments\E14_stable_teacher\generate.ps1
```

依次生成train16,000及固定dev_monitor500，仅后者用于验证，不训练学生，不读取local_test。CPU两线程；训练库累计24小时、开发监测库累计4小时保护上限。预计数小时，实际取决于救援频率，磁盘预留2GB；保护上限不是收敛判据。布局与结构标签存`labels/*.npz`，包含修正后的`input_areas`、任务版本和schema；源码、权重及成员哈希随库保存。

目录：`runs/teacher_train16000_v1`和`runs/teacher_devmonitor500_v1`。`progress.json`记录PID、完成数、错误、累计耗时和最后更新时间；`records.jsonl`逐例持久化，`results.json`每25例及停止时保存完整状态。新的标签若硬检查失败或Q>2，保留拒绝候选并停止，不把它静默放入新库。

停止当前训练库：

```powershell
New-Item -ItemType File -Path experiments/E14_stable_teacher/runs/teacher_train16000_v1/STOP
```

当前例完成后停止，后续开发阶段不启动。移除这个明确的STOP文件后，使用`generate.ps1 -Resume`恢复；累计时间、已完成成员和标签哈希保留，已完成阶段不会重复生成。源码、任务身份、成员或预算不一致时拒绝恢复；存在失败成员时也拒绝跳过失败继续。需修复时开新版本，不能覆盖旧失败事实。

后台启动记录在本实验`runs/background_launch.json`，包括父进程PID、标准输出/错误日志和启动命令。后台生成没有完成全库时，只能报告“已启动”，不能声称完整教师库已经通过质量审计。
