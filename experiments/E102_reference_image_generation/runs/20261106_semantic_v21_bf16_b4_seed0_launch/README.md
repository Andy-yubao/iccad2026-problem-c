# BF16＋batch4工程迁移续训

用户2026-11-06明确授权：使用刚讨论的BF16混合精度／batch4准备续训，准备好后后台启动。仅此一个迁移运行；不增加batch8或新候选，不创建自动唤醒，不读取测试。

## 继承与变化

从原semantic正式运行最后完整12297步latest继承模型、FP32 Adam各状态／步数、CPU/CUDA／遮罩RNG、16000成员排列、cursor8594、seen24594、调度及停滞计数。旧latest／best／源码快照和两级STOP保持原样。短检更新只用于检查，不作为续训来源。

原累计11778.767秒保留在原总28800秒预算内，剩余约17021秒（4.73小时）；不另开8小时。保持原LR2e-4、Loss公式、归一化、双视图权重、图像尺度、冻结成员与数据顺序。

网络前向采用CUDA BF16 autocast；输出转回FP32后，在autocast外计算包括概率BCE的全部Loss。模型主参数、梯度和Adam仍FP32，不使用FP16或GradScaler。保留有限性检查、原clip_grad=1及参数门禁。

batch2改为4是明确的优化路径变更。成员顺序／游标不变，成批分组及遮罩RNG的消费方式随批量改变，因此不宣称逐位等价续训。更新计数从12297延续，之后一次通常处理4个样本；不能将迁移前后相同步数当作相同曝光量。

监测以原每4000个seen成员的节奏继续，下一阈值seen28000，触发在下一完整批次边界（至多差3个样本），每轮结束也监测。原调度／早停坏事件计数均继承，未重置。

## 数值基线与best

为避免把精度变化当作质量提升，在BF16下对原已冻结best做一次固定dev500重新计值，再对继承latest做固定train/dev500基线。此阶段不更新参数、不推进调度／早停计数；原调度和significant-best的历史标量按同一best权重在新旧精度下的差值校准，计数／LR保持。新best只比较新精度下的同一Loss，原FP32报告和best原件不改写。

实际续训仍从latest权重和对应Adam开始，校准期间临时推理best不改变优化轨迹来源，也不按Q选权重。新latest和完整状态保存到独立run；重复启动不会覆盖旧运行。

## 已完成短检

只使用原cursor后紧接着的四个train成员，一次真实BF16 batch4更新；FP32参考对同四例按microbatch2只前向计算。Loss和参数有限，Loss相对差0.00093644（约0.094%），低于事先登记2%门禁，FP32主参数／Adam保持。峰值allocated13730MiB（约13.41GiB）。见[原始结果](engineering_check4/result.json)。单次冷启动更新耗时不代表稳态吞吐；实际速度看正式续训日志。

模型／Loss／数据来源哈希在migration.json中核对不变。没有预先做断点恢复测试，没有dev调系数。

## 入口

- 新运行：`../20261106_semantic_v21_bf16_b4_seed0/`
- [实时启动状态](state.json) · [固定方案](plan.json) · [迁移身份](migration.json)
- [标准输出](train.stdout.log) · [错误输出](train.stderr.log)
- 固定dev100仍只用全MASK直接生成，在`visualization/E102_reference_image_generation/20261106_semantic_v21_bf16_b4_seed0/`；迁移后100次更新预览，随后每次Loss-best更新。

停止时在新run或本launch写STOP；旧运行STOP不清除、不参与新目录的运行控制。新任务失败即停，不自动重试。
