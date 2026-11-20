# 2026-11-06：心跳文件替换失败修复

原训练第1192次更新后，写heartbeat.json时发生WinError 5。最后成功发布的心跳为1191步，最后完整latest为1100步／cursor8800；退出由日志文件替换错误触发，已有记录未显示数值门禁失败。不能仅据此确定哪个进程占用了文件。

原错误、心跳、state、完整latest、源码身份、原源码快照和启动日志均已复制保存在本目录。原件保留，不把旧error.json改写成成功状态。

## 修复

仅修改train.py并增加file_io.py，不改模型、Loss、成员、公开输入、batch8、FP32、LR或数值门禁。

- Windows访问／共享冲突5/32/33时进行有限时长重试；其他错误仍抛出。
- heartbeat及描述性state发布受持续占用时记录警告，保留已提交更新并继续。完整checkpoint写入仍是必要操作，不悄悄忽略失败。
- 心跳最多每秒发布一次，遇持续占用暂停十秒再试；每50步另写updates.jsonl及标准输出，便于读日志。
- 显式核对迁移前后全部源码身份，允许变化仅为上述两个文件；不关闭身份检查。

## 针对性验证

使用Windows CreateFileW实际持有允许读写但禁止删除共享的句柄，复现WinError 5。短暂锁释放后有限重试成功；持续占用时advisory写入返回失败而不抛出训练异常；释放后下一次写入成功。见io_check.json。没有额外使用布局题，没有进行预防性断点恢复测试。

## 实际恢复

用户本次明确要求修复当前运行，在原授权单次训练的剩余预算内实际恢复。继承1100步latest内模型、Adam、RNG、排列／游标、调度及停滞状态。1192与1100之间的92次更新需重新计算，不声称从1192步无损恢复；原完整latest在此目录保留。未入checkpoint但已耗时20.304156780秒加入累计预算；暂停排查期间不计训练时间。迁移细节见migration.json。

当前运行日志看原launch目录的train_resume_iofix.stdout.log和train_resume_iofix.stderr.log，原train.stderr.log只保留第一次失败现场。训练目录updates.jsonl提供每50步追加记录，heartbeat.json为最近成功发布的心跳。原state1100和旧error.json不能单独当作恢复后的实时状态。
