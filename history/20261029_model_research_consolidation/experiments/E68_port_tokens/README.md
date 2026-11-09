# E68 v2：完整端口组合方案

用户2026-10-28要求优先尝试合理组合，明显进步后再做回溯消融。本版只准备一个seed0候选；原均值对照不再安排正式训练。旧版从未启动，源码/计划/短查保存在[runs/preparation_v1/](runs/preparation_v1/)，不改写旧记录。完整输入审计见[input_audit.md](input_audit.md)，执行范围见[plan.json](plan.json)。

## 本次组合

1. 原96×12矩形主干的节点输入从17维改为14维，完全移除端口总权重、端口平均x/y三个摘要通道。原冻结cache仍保留17维，但在网络边界裁成14维；推理路径也只传14维，模型拒绝17维输入。
2. 每层24维端口分支读取每条正权连接的完整端口坐标与权重、相对当前模块的位移。重复同一端口的边先合并；没有TopK或连接截断。模块无直连时，该分支为零，由原全局模块attention和图消息传递其他模块读取的信息。
3. 修正原先纯归一化注意力会抵消整体权重缩放的问题。每条权重直接进入token；采用门控加权求和与asinh幅度变换，保留绝对强度和连接数量影响。没有额外提供人工总权重/均值特征。

组合总体与保存的E64/E65及同例E42草稿比较，不把收益归因到单一组件。新增投影保持零初始化；输入投影复制原初始化的前14列，主干其余随机初始化保留。不是与旧17维输入完全相同的初始函数。暂不增加传播标记、联合避碰模块或实际线长Loss。

## 固定训练与评价

原八项G及v5 train16000、固定train/dev500、batch32、FP32、AdamW/clip1/dropout0.05和四档LR/早停不变。仅seed0，不做多种子。最多3小时或200轮，恢复保留预算/优化器/调度/RNG/游标；测试封存。旧共享cache只读，额外公开端口在内存打包。显存不足不静默改batch、精度或机制。

唯一dev G best完成raw dev2000、原flash500及同100例reference/current/E42无评分图。报告低端口/稀疏子群、无直连模块误差、近邻相对误差及次序翻转。Q只辅助，不据此重选checkpoint。历史基线只读复用；若只改善富端口组，明确弱覆盖问题仍未解决。

## 执行

本轮只准备及同五个训练成员的实现短查，助手不启动正式训练，不创建自动唤醒。PowerShell：

```powershell
Set-Location 'E:\project\ICCAD\2026\problemC'
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E68_port_tokens.pipeline --resume
```

命令只启动本候选，随后评价、作图、汇总为runs/review.md。输出为runs/20261028_full_ports_nosummary_v2_seed0，日志runs/full_ports_train.log；state.json记录阶段。原两组命令已被本版替代。

源码需通过[runs/preflight/results.json](runs/preflight/results.json)且身份一致。修改源码后须重新短查；已有正式训练不能换源码继续。命令预览或重新短查：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -B -m experiments.E68_port_tokens.pipeline --dry-run
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E68_port_tokens.preflight
```

Ctrl+C或创建runs/STOP可在保存边界停止；恢复先移除自己创建的STOP，再执行同一pipeline --resume。best/latest分开保留，不清零累计预算。本候选完成即结束，不恢复旧长期研究。
