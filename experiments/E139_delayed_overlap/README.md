# E139：延后重叠监督，保留后段完整优化

2026-11-15 用户要求仅准备实验和训练入口，**现在不启动训练**。E138 隐空间重规划本轮已宣告失败，新实验回到未改动的 E119 架构，只改变分层监督。目标是让前段可以经过重叠状态调整拓扑，再由后段处理重叠并优化弱相对位置；这是假设，不是已经获得的能力。

| 状态 | 监督目标 | 固定权重 |
|---|---|---:|
| S2 | 完整 Loss 去掉全部重叠项 | .05 |
| S6 | 完整 Loss 去掉全部重叠项 | .05 |
| S8 | 原完整 Loss | .05 |
| S10 | 原完整 Loss | .05 |
| S12 | 原完整 Loss | .8 |

`J = .05*(L2无重叠 + L6无重叠 + L8完整 + L10完整) + .8*L12完整`。前四项均分 .2，不增加 S4/S0 监督。S2/S6 删除的是直接碰撞组 `A*(penetration + module_load_mean + module_max_excess10)`，A=.08975162861857286；其余参考位置/形状、线长、紧凑、分组、贴边、F/P 尺寸、P 位置项的系数与归一化均保持 E134 public-priority-v1。原始碰撞量仍作为诊断保存；它不进入 S2/S6 自身的总 Loss。后层完整 Loss 仍通过前层反传，前段并非完全不受重叠梯度影响。几何反馈、空间偏置、共享预测头均不改，无隐空间重规划、扰动恢复分支或中间状态截断。

E119 原结构 712275 参数，scratch seed0。延续本轮固定 36k、原 train500/dev500、FP32/batch32、增强、AdamW、LR/早停与 Adam 前数值门禁，每组累计最多 4h/200 轮。复用兼容持久缓存，不重抽数据、不重建预处理。全模块预测，包括 F/P；不覆盖草稿、不使用 MIB、不读 test。反传、固定监测、best、LR、早停均使用上面同一个 J；共同末层完整 Loss 仅作诊断。

复用 E134 每三层及每两层既有结果，不新增对照训练。相对它们同时改变了监督层位和碰撞监督分布，单组结果不能单独归因于去重叠，更不能据单 seed 宣称可靠突破。后续应分别核对强关系修复、弱关系/任务质量、草稿碰撞、最终硬合法、Q 尾部和完整耗时。

## 训练入口（留给之后明确启动）

在项目根目录运行以下命令会激活新队列；本次准备不执行：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E139_delayed_overlap.run_queue --start
```

沿用本轮队列：训练成功结束 → 冻结唯一 own-Loss-best → 原固定 dev500、原 E98.depth_budget.v1 后处理 → 汇总。Q 不参与选模，不扫描后处理预算。后处理和评价代码只在将来显式启动队列后使用，本次不运行。错误阻断队列，不自动重试、不追加候选或 seed。

前台按一次 Ctrl+C 后等待安全暂停，恢复命令：

```powershell
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E139_delayed_overlap.run_queue --resume
```

恢复 model/Adam/RNG、排列/游标/seen、LR/早停、投影状态和累计预算；不清零 4h。评价按连续已提交例续跑。STOP/QUEUE_STOP 不自动删除；历史实验不恢复。

`plan.json`、`loss_config.py` 冻结配置；`provenance.json` 记录独立源码及来源；`preflight.json` 记录五训练例准备检查（检查模型丢弃，正式启动重新 scratch 初始化）。续训/控制器/评价路径复用 E134 已验证实现并检查身份替换后的对应源码；不重复完整中断模拟。准备完成时 `authorization.json` 保持 training_enabled=false，`queue_state.json` 为 prepared，正式 `runs/` 不存在。
