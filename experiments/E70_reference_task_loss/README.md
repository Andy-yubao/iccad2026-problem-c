# E70：新任务Loss＋完整端口

2026-10-28按用户反馈完成[外框与填空排查](runs/20261028_bbox_void_audit/report.md)：同500例外框较E68平均扩大2.742%；精确空白比例较E68下降、较E65略升，同100图较E42外框扩大2.579%。五例向内压缩1%虽改善外框/空白但新Loss均上升，紧凑项中心梯度仅覆盖2个极值模块，提示目标取舍与内部协调填空缺口。用户不接受以扩大外框换避碰；不续训，不升级，测试封存。

2026-10-28训练及终末评价已完成：[成果复核](runs/inspection.md)。74轮正常早停、best71，dev2000草稿交面积3.820%（较E68下降54.66%），但位置/近邻误差与flash加权Q退化；保留避碰候选，不升级E42。下文启动说明保留为复现入口，不自动续跑；测试封存。

按用户要求准备一个组合候选：沿用E68 v2的完整端口模型（14维节点输入、不重复输入端口总权重/均值、逐边权重门控求和），改用新固定Loss。原E68及E42/E64/E65权重和证据不变。先验证组合，明显改善后再回溯消融；本轮不启动正式训练或自动唤醒。

## 固定目标

reference仅用于位置和log长宽比。其余八项只读公开任务和预测矩形：穿透、真实交面积、实际b2b/p2b线长、公开面积归一化紧凑度、允许自主拓扑的MST接触连通、公开边界、MIB同形、参数域保护。详见[公式](loss_design.md)、[固定系数](proposal.json)、[执行计划](plan.json)、[输入审计](input_audit.md)。最初设计保存于[runs/design_v1/](runs/design_v1/)。

原G只在监测时另算为old_G，不加入新总Loss，不参与best/LR/早停。不能直接比较新总Loss与旧模型G。各项权重从第一步完整使用，全程不改变；未用开发集或Q调权。五例检查用于数值/梯度量级和实现正确性，不证明系数最优。

分组MST使用固定N次GPU Prim森林，拓扑由当前预测代价detached选择，梯度只通过选中的接触代价；无CPU选树或隐式切换引擎。相同公式用于训练与监测。接触要求1%短边的正长度投影，角接触不当作满足。硬面积、固定WH、预放置XYWH仍通过解码保证，有限碰撞Loss不保证零重叠。

## 训练与评价

从头训练seed0，与E68 v2使用相同模型初始化、v5 train16000/固定train-dev500、batch32/FP32、AdamW decay3e-4、FFN dropout.05、clip1和四档LR/早停规则；不做多种子，也不从E68 best续训。最多3小时或200轮，恢复不清零优化器/调度/RNG/游标/已见成员/累计预算。

唯一新Loss开发best完成raw dev2000、原flash500和同100例reference/current/E42无评分图，随后生成runs/review.md及review.json。旧G作为共同诊断，重点看实际重叠、无直连模块位置与近邻错误、弱端口且稀疏子群和用户草稿观感。Q仅辅助，不重选checkpoint，测试封存。历史E64/E65/E68读既有证据，不重复训练/评价。

## 启动

先通过[五例预检](runs/preflight/results.json)并核对源码身份。助手本轮只执行短查，不代为开启正式运行。PowerShell：

```powershell
Set-Location 'E:\project\ICCAD\2026\problemC'
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E70_reference_task_loss.pipeline --resume
```

命令只运行本候选及上述评价/作图。输出runs/20261028_full_ports_taskloss_v1_seed0，日志runs/full_ports_train.log及full_ports_final.log；state.json为控制器阶段。命令预览：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -B -m experiments.E70_reference_task_loss.pipeline --dry-run
```

修改源码后重新短查：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E70_reference_task_loss.preflight
```

Ctrl+C或创建runs/STOP会在保存边界安全停止。恢复先移除自己创建的STOP，再用同一pipeline --resume。开始后不换Loss/源码/数据，不覆盖旧运行。完成即结束，不自动继续其他实验。
