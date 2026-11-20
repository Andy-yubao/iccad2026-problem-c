# E131：P0 完整关系输入编码对照

2026-11-12 用户只授权准备，不启动训练，本次不运行后处理。

两组共用 400→128→64 入口编码：原16维公开特征、自身地址128维、完整归一化连接行128维、含自身的同组成员行128维。仅入口编码一次；临时地址按固定随机流每样本每轮重命名，监测和推理用固定身份地址。地址不具有布局语义，任意地址重命名的输出不变性没有数学保证。

| arm | 主干与标准读出 | 每层额外通路 | 参数 |
|---|---|---|---:|
| full | 64×4、4头、SwiGLU128；全局均值拼接128→64→3 | 关系偏置＋连接/G组隐藏聚合 | 268387 |
| pure | 相同 | 无；块只接收h、valid | 235331 |

共同初始化的编码器、QKV/FFN/归一化及读出张量完全相同；额外分支用独立CPU随机流初始化，不改变公共初始化或CUDA RNG。不称等参数/等算力，也不将联合删除解释为分别确认两项机制的贡献。

无中间草稿、预测几何反馈、辅助读出或F/P通道覆盖。独立末端公开硬规格优先Loss：原reference/G/B/碰撞/线长不变，公开尺寸与P中心项各权重1。所有模块仍预测位置和形状。

准备采用冻结16k子集内原E116五个训练成员，复用已完整的原始面积/无MIB/全模块预测CPU缓存；不重新生成样本或静态特征，不读dev/test张量。完整训练入口已准备，但授权开关false、允许arm为空且本实验STOP保留。准备只检查前后向与CUDA图，零Adam更新，不创建run或checkpoint。

预备正式设置：seed0、冻结train16k/子集train500/原dev500、FP32batch32、原轴交换与Dropout、AdamW/原数值及0.5%相对更新门禁。统一四档LR，14轮保护、patience6、改善阈值max(1e-5,|best|×1e-4)，最低档至少8完整轮且8次无改善；best/LR/早停使用同一固定末端目标与改善规则。每组上限4h或200轮，到预算仍改善则标记未测到上限；不自动续训。与旧E116/E119的调度不同，不称单因素历史消融。

入口（现在会在运行创建、CUDA初始化、数据读取或优化器更新前拒绝）：

```powershell
.\.conda\envs\iccad2026\python.exe -B -m experiments.E131_relation_input.train --arm full
.\.conda\envs\iccad2026\python.exe -B -m experiments.E131_relation_input.train --arm pure
```

准备检查：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E131_relation_input.inspect
```

尚无启动授权；仅移除STOP也不能启动。将来启动须明确授权并核对源码与门禁；每组只保存自己的best/latest和Loss日志后停止。无自动队列、后处理、Q、test或新增图；错误/用户停止不自动重试。latest恢复保留Adam、RNG、排列/游标、调度、投影及累计预算，准备不测试恢复。

主配置plan.json，授权authorization.json，短检preflight.json，源码来源provenance.json。算法本地独立维护，仅共享datasets元数据辅助与原CPU缓存。不恢复任何归档实验，不改变E119研究/E101交付或旧STOP。
