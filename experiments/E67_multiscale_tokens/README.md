# E67：冻结G下的多尺度分组与全局对照

2026-10-28用户明确中止训练：group_tokens已完成训练及唯一best评价；global_tokens于第37轮边界18,500步安全停止，保留best36和可恢复latest，尚未完成本组终末评价。控制器退出，长期自动唤醒删除，E68/增强未启动，不自动恢复或补评。[总任务终止记录](../E65_geometry_messages/long_task.json)。下文为原预登记方案，不构成继续执行授权。

两组五例检查通过，已后台启动group_tokens seed0并核对实际更新；随后自动运行global_tokens seed0。参数量均1,659,603；冻结Loss/梯度与CUDA/eager实际Adam差值为0，含Dropout断点恢复逐位一致，MIB中心独立性与成员池守恒通过。实时进度见state和日志。

长期研究第三阶段，全部seed0、不做多种子，用户授权后台训练和每小时继续。E66当前增量/共享实现未获益，停止扩展并保留证据。E67独立回到原E64 96×12矩形主干，增加多尺度状态，不叠加E65消息或E66增量解码。[状态](state.json) · [长期计划](../E65_geometry_messages/long_task.md) · [累计预算](../E65_geometry_messages/long_task.json)。

每层有24维grouping状态与一个global状态，按公开分组聚合模块隐藏特征，通过4头注意力/FFN交换粗粒度信息，再广播回模块。token状态跨12层保留，原模块间交互仍在。另一条MIB状态只读取当前可变或公开固定log宽高比与公开面积，独立更新，不接收中心或group/global token，不强制MIB模块位置聚集；其hidden广播可以间接影响位置，不能称完全无位置影响。

| 组 | 新分支的聚合/广播 | 保留内容 |
|---|---|---|
| group_tokens | 真实公开grouping/MIB成员 | 原主干与全局状态 |
| global_tokens | 所有有该约束的成员均匀混合到同数token | 参数、初始化、形状、正约束标志、组数及原主干关系 |

因此对照移除的是新分支的真实成员信息，并未让整个模型失去分组信息。token槽取正组中首个有效模块，不使用组ID数值embedding；ID重命名不改变结构。固定最多120个槽，padding无效。新分支输出零初始化、残差系数0.1，fork_rng保留原主干初始化及随机状态；初始输出与原E64逐位相同。参数数由预检/正式config记录。

原E64八项G源文件字节保持，仅最终输出、固定完整权重、无额外token Loss；训练/监测/best/LR/早停同一G。v5 train16000原顺序、train/dev500、batch32、FP32、dropout.05、AdamW decay3e-4、clip1、原四档LR和15轮保护期不变。每组3小时/200完整轮，恢复和失败计入全任务10次/18小时与72小时日历保护，不重置。没有新数据缓存或测试读取。

同五train案例验证冻结G/梯度、原E64初始化等价、pool成员/padding守恒、真实分组影响、MIB输入不依赖中心、实际CUDA/eager Adam、非有限值保护、含Dropout断点恢复逐位一致。新实现仅在预检通过及源码身份一致时启动。[模型](model.py) · [预登记](plan.json) · [来源与保护哈希](provenance.json) · [预检](runs/20261028_preflight/results.json)。

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E67_multiscale_tokens.preflight
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E67_multiscale_tokens.pipeline --resume
```

项目根目录，用独立`.conda/envs/iccad2026`。持续exec会话保留父控制器，依次group与global组训练/唯一best raw2000/冻结flash500/同100图。CPU评价2进程各2线程、图4进程各1线程。输出`runs/20261028_<arm>_seed0/`，日志`runs/<arm>_train.log`；完成报告[runs/review.md](runs/review.md)、[review.json](runs/review.json)。Q只辅助，不重选/自动升级，用户图评优先，原E42/E64/E65/E66权重保护。

停止创建本实验`runs/STOP`；移除自己创建的STOP后同pipeline --resume恢复，保留优化器、调度、RNG、游标、已见成员和累计预算。Loss/数据/源码变化不绕过身份校验。阶段完成后每小时唤醒复核结果并继续E68端口信息，全部单种子自适应开发证据，测试封存。
