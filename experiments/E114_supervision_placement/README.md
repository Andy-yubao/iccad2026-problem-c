# E114：小模型监督位置对照

2026-11-07新增：用户明确要求“在后台启动训练”。三组按terminal→dense→staged串行，各原4小时/200轮上限；错误/用户停止不推进，不自动评价Q、不读取测试、不恢复旧实验。实时状态见queue_state.json与各run heartbeat。准备阶段原件在launch_registration_20261107；以下“不启动/开关关闭”是此前准备阶段记录，已被本次明确授权覆盖。启动不改变模型、Loss、数据、数值路径或预算。

用户2026-11-07要求准备入口，不启动训练。`authorization.json`的training_enabled=false，实验STOP保留；没有后台队列，没有自动后处理/评价。旧实验STOP、权重、数据索引不改。

三组均64维、12独立图块、共享64→64→3预测头，小几何MLP18→24→4。按新协议删除MIB输入/聚合/偏置通道，总参数612307（全部可训练）；未删除MIB时约66万，差异不是新增候选。只改变监督位置：

|arm|固定目标|
|---|---|
|terminal|L12|
|dense|0.8 L12 + 0.2 mean(L1..L11)|
|staged|0.8 L12 + 0.2 L4|

每个L为同一完整逐例Loss，先层加权再样本等权平均。仅计算有非零权重位置的完整Loss；所有12层坐标反馈保持。没有冻结、detach或强制逐层单调改善；最终层梯度贯通所有层。密集和阶段监督辅助权重总量均.2，不搜索系数；历史A原权重不同。

## 数据及预测边界

采用已准备random_reference40k.v1的train36000和固定train/dev各500监测；新test2000仍封存。直接验证NPZ校验和、读取原始面积和公开规格，不用旧16k/MIB平均面积缓存。忽略MIB特征、Loss及软归一化分母；固定/预放置规格仅作条件输入，全模块位置/形状参与监督，草稿解码不覆盖规定尺寸/坐标。面积保持由原始公开面积解码。后处理才负责满足硬约束，最终原规格不变。

这是新任务，不能与旧A/E105/E74分数直接作控制比较；三新组相互可比。后续Q标为不计MIB的本地研究Q。当前没有启动Q或后处理；终末评价应冻结各自own-Loss best后在共同后处理上进行，不能按Q回选checkpoint。

## 使用入口

在项目根目录，用独立环境运行：

```powershell
.\.conda\envs\iccad2026\python.exe -B -m experiments.E114_supervision_placement.inspect
.\.conda\envs\iccad2026\python.exe -B -m experiments.E114_supervision_placement.train --arm terminal
.\.conda\envs\iccad2026\python.exe -B -m experiments.E114_supervision_placement.train --arm dense
.\.conda\envs\iccad2026\python.exe -B -m experiments.E114_supervision_placement.train --arm staged
```

后三条是未来训练入口，目前均会在数据/GPU计算/run创建前明确拒绝。不能仅移除STOP：还需要用户明确启动授权后登记authorization.json，并保持实施检查源码身份匹配。本次不修改开关。`--resume`仅恢复同arm完整latest，不回载best续训，不清零预算；错误/已完成运行不能直接恢复。

每组上限4小时或200轮；14轮保护后原四档LR/早停。batch32、FP32、同seed初始化/数据顺序/轴交换/Dropout策略和更新门禁。训练数据CPU驻留、pinned批传输，固定120模块填充、Loss端口上限由实际训练/监测公开输入决定；不全库驻留GPU。不预称速度或能达到收敛；4小时上限不是等更新量。

best/latest分开，完整保存Adam/RNG/排列游标/seen/调度/步长保护/累计预算。初始和每轮固定train/dev500用各组同训练目标监测，另报共同final_total；真实steps.jsonl记录训练吞吐，progress记录含监测保存的整轮时间。无自动下一组。

## 本次实施检查

`inspect`只用登记五个train成员，三组同权重输出、输入标签隔离、MIB不影响特征、全模块预测、层权重、梯度、FP32 CUDA图捕获/重放及数值门禁检查。零Adam更新，不预演恢复，不读取dev/test。结果写preflight.json，正式训练仍禁用。五例不是泛化或完整epoch性能证据。训练后的逐层图应走公共visualization，同五例同尺度；本次没有训练后输出。
