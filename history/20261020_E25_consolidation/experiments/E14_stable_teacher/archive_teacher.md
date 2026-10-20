# 25,000例完整布局库

当前入口：`generate_archive.ps1`。完整布局库与教师筛选清单分开保存。模型、权重、MIB变换与后处理保持原版本；单GPU float32草稿、4个CPU后处理进程、主进程评分及单点写库。旧库原件和历史记录保留，不恢复旧入口。

## 运行

先在旧终端结束旧生成进程（Ctrl+C；若只是暂停控制台，应先解除暂停再结束）。暂停的进程仍可能占用锁和硬件。新库通过SQLite一致性快照迁移已提交成员及不可变标签，不篡改原库，也不会包含原库在快照之后的新增结果。

在项目根目录运行：

```powershell
.\experiments\E14_stable_teacher\generate_archive.ps1
```

新库：`runs/teacher_archive25000_v1/`。本机已迁移4905份已接纳布局，3199个历史DROP登记为优先重算。历史DROP不论原Q、不论旧文件是否仍在，全部重算；保留原记录在historical_records表，不用新结果覆盖历史证据。

目标是25000个不同成员的持久布局：已有4905例计入，重算历史DROP只补齐这些成员的完整新结果，不重复增加样本数。启动前待计算20095例，其中3199例历史DROP重算、16896例此前未评分成员。已登记但没有完成结果的在途任务按未完成处理。真实派发次数（含失败、重试）单独累计，不冒充不同样本数。

新增完整库的六层名额为3886/5327/5308/4892/2978/2609，按原train16000模块数比例扩大至25000。优先完成全部历史DROP，再按层补足。新候选仍只来自split_v1原train分片，沿用MIB兼容过滤；不读取dev/test数据。完成25000即停止，不为补齐教师名额自动超出25000。

默认新库累计预算48小时或60000次派发，任一触顶停止；预计为数小时到十余小时，受历史尾部任务和兼容候选扫描影响，预留5–10GB磁盘。需要时可显式扩大资源预算，但成员总目标仍是25000：

```powershell
.\experiments\E14_stable_teacher\generate_archive.ps1 -MaxSeconds 345600 -MaxAttempts 120000
```

## 保存内容及筛选

所有返回的布局都落盘，不因Q过高或硬检查失败删除。新计算NPZ保存完整positions、派生input_areas、b2b/p2b/pins、constraints、specs、任务版本及评分/来源record_json；硬合法时还保存训练需要的结构标签。新评分/编码错误同样保留坐标、输入和错误信息，并禁止进入教师集。算法异常导致没有返回布局时记录错误并停止，成员仍可恢复重试，不能捏造一个完成样本。

迁移的已有合格NPZ保持原始文件和校验值，已含完整positions、结构标签、派生input_areas与版本；其余公开输入按记录中的冻结源成员恢复，无需重新求解。旧文件本身不改写。所有新结果具有唯一attempt文件名，恢复不会覆盖此前完成的尝试。文件原子落盘后再提交SQLite；若中断恰发生在两者之间，恢复从NPZ内record_json补交，避免重复求解。正常STOP会排空在途任务；Ctrl+C或其他任务失败时，也会保存已经完成并返回的在途结果，取消未开始任务。

教师清单默认Q≤1.06且硬合法；每层按(Q,成员ID)从优到劣选至原16000目标配额，多余候选仍在完整库中。25000例全部计算完毕后若某层合格样本不足，清单明确报告缺额，不降低标准或复制样本。Q仍是原参考分母下的中性时间诊断，不是学生成绩或正式比赛分数。

输出：

- `bank.sqlite3`：断点权威状态、成员、历史记录、尝试与评分。
- `layouts/`：所有完整布局，包含不进入教师集的样本。
- `archive_manifest.jsonl`：所有已存布局的索引。
- `teacher.json`：当前筛选后的教师成员清单。
- `progress.json`：不同布局数、历史DROP剩余数、真实派发次数、教师缺额。
- `source.zip`：本轮实际源码快照；`sessions.jsonl`记录命令和预算。

进度每25次提交更新；完整索引/教师清单每250次及退出时导出，因此运行中的teacher.json可能稍滞后于SQLite。恢复会核验已保存文件哈希和源码/协议身份。

以后调整阈值只重新筛选，不重算、不删除布局（在生成停止时执行）：

```powershell
.\.conda\envs\iccad2026\python.exe -B -m experiments.E14_stable_teacher.archive_bank select --quality-max 1.04
```

例如改回1.06，只将参数改为1.06。生成入口仍按当前协议默认1.06导出；最终需要其他阈值时，生成结束后执行select。

## 停止与恢复

```powershell
New-Item -ItemType File -Path .\experiments\E14_stable_teacher\runs\teacher_archive25000_v1\STOP -Force
```

当前在途结果保存后停止。继续时：

```powershell
Remove-Item -LiteralPath .\experiments\E14_stable_teacher\runs\teacher_archive25000_v1\STOP
.\experiments\E14_stable_teacher\generate_archive.ps1
```

未建立STOP文件时，重新执行生成命令即可自动恢复。不要删除layouts、SQLite或源码快照；只保留teacher.json不足以恢复。

本机已完成5例历史DROP重算功能检查，全部布局已保存（其中4例Q>1.06未入教师集）；当前完整布局4910例，历史DROP待重算3194例，教师清单4906例。已核验全部文件哈希和模块数、训练分区成员、阈值重新筛选不删除布局、文件先落盘/数据库未提交时的恢复、硬非法坐标仍能保存及排除教师、STOP零新增尝试。详细证据在`runs/20261019_archive25000_smoke5/summary.json`。没有启动长计算。
