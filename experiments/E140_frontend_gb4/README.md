# E140：仅 S2/S6 的 grouping、boundary 各增强4倍

2026-11-15 用户要求准备一组实验，以已完成 E139 为对照。本次**仅准备，不启动正式训练**。参考关系图相似度已获用户认可，留待后续与基础 reference 配合；本组不引入该指标或初始头/拓扑架构改动。

|状态|监督目标|外层固定权重|
|---|---|---:|
|S2|E139 无重叠 Loss，grouping/boundary 各×4|.05|
|S6|E139 无重叠 Loss，grouping/boundary 各×4|.05|
|S8|E139 原完整 Loss|.05|
|S10|E139 原完整 Loss|.05|
|S12|E139 原完整 Loss|.8|

Gk/Bk 表示 E139 第k层**已经乘过原系数**的 grouping/boundary 贡献，因此：

`J_E140 = J_E139 + .15*(G2+B2+G6+B6)`。

四倍作用于原项，不改变其分组/边界聚合和归一化，不重新归一总 Loss；不将外层 .05 改成 .2。S2/S6 的碰撞组仍全部删除；基础 reference、线长/紧凑、F/P尺寸、P位置及所有后层项保持。未监督层仍接受后续梯度。

模型、初始化与数值路径延续 E139：原 E119 两个全局块＋12层几何主干，712275参数，scratch seed0，与 E139 逐张量相同初始值；固定36k、原train500/dev500、FP32/batch32、增强、AdamW/LR/早停、Adam前数值门禁和4h/200轮累计预算。不从 E139 best42 续训，不重训对照，不扩展候选/seed。复用现有兼容CPU缓存，不重建静态预处理。

对照冻结为 E139 best42，checkpoint SHA256 `b50a4e0820c25bc07661797efaae4b6a7c7c97d800688104d57aeab9989835e6`；对照原件与哈希在 `plan.json` 和 `provenance.json`。相同成员、初始化与预算，但不同 own-Loss 可能造成不同早停，单seed仍不证明可靠收益。自己的聚合Loss不跨组排名，公共末层完整Loss保持可比。

## 启动入口（本次不执行）

项目根目录 PowerShell：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E140_frontend_gb4.run_queue --start
```

显式启动后只训练本组；成功后冻结唯一 own-Loss-best，再自动原dev500、原 `E98.depth_budget.v1` 后处理、原面积/无MIB研究Q及独立硬检查，最后与E139已存结果对比。不按Q选权重、不扫描预算、不读test。比较报告另列分组/贴边违规及全部软约束满足题数，不将未满足boundary自动归为题目矛盾。

前台按一次 Ctrl+C，等待安全保存后恢复：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E140_frontend_gb4.run_queue --resume
```

沿用已验证的 E139 控制器与训练/评价续跑引擎，保存模型、Adam、RNG、排列/游标/seen、调度/早停、投影状态及累计预算。错误阻断队列、不自动重试；STOP/QUEUE_STOP不自动删除，历史任务不恢复。

`preflight.json`记录原五训练例的Loss/梯度与一次可丢弃受保护更新；检查模型丢弃，正式启动重新scratch初始化。续训证明按对应源码一致性继承，不额外模拟完整中断。准备完成时授权开关关闭，队列prepared，正式runs目录不存在。历史 E139 源码、权重、数据和结果不改。
