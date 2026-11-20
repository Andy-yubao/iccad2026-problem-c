# E135：E119 的 S0 监督 A/B/C

用户于 2026-11-14 指定三组，并授权准备通过后后台训练、自动后处理。只研究 E119，不重训已有对照，不恢复 E120。

| 组 | S0 Loss | S0 | S3 | S6 | S9 | S12 |
|---|---|---:|---:|---:|---:|---:|
| A | 相同完整新 Loss | .05 | .05 | .05 | .05 | .8 |
| B | 相同完整新 Loss | 1/30 | 2/45 | 1/18 | 1/15 | .8 |
| C | 下述 F | .05 | .05 | .05 | .05 | .8 |

B 的辅助权重为 3:4:5:6，随层号线性增加，S9 为 S0 的两倍；辅助总权重均为 .2。完整新 Loss 沿用 public-priority-v1，包含 F/P 尺寸、P 位置项。所有权重全程固定。

C 的 S0 使用 `F = 中心容差 + .05*自由形状 + F/P尺寸 + P位置`：

- 非 P 模块中心，每个坐标轴允许偏离 reference 中心 `.25*sqrt(模块面积)`；超出部分除以 sqrt(面积) 后做 Huber（beta=.1），逐例在有效非 P 模块的两个坐标上取均值。
- 自由形状项是非 F/P 模块的原始 log 宽高比与 reference 的平方差均值。
- F/P 尺寸和 P 位置严格复用完整新 Loss 的公开规格项，系数各为 1。各子集分别归一化，空子集为零。
- S0 不含重叠、紧凑、连线、G/B 项；S3/S6/S9/S12 均保留完整新 Loss。reference 只进入 Loss，不进入模型。

S0 是已有两个前端全局块之后、首个几何块之前的共享头输出。只多返回这个已有状态，不增加头、不改变参数或 S1–S12 的前向计算。全模块预测，不覆盖 F/P，不使用 MIB。

沿用 E134 E119 每三层配置：36k 固定训练集、固定 train/dev500、scratch seed0、712275 参数、FP32/batch32、原增强、AdamW/LR/早停/数值门禁，每组 4 小时或 200 轮上限。已有 E134 stride3 best63 作为历史对照复用。各组 own-Loss 用于反传、监测、best、LR 和早停；不同目标的总值不直接横比，共同末层完整新/旧 Loss 用于诊断。

队列依次执行 **A训练→冻结唯一 own-Loss-best→原 dev500 后处理→B→C**。沿用原 `E98.depth_budget.v1` 预算与原始面积，无 MIB Q，原规格独立硬检查和完整逐例耗时；不按 Q 选模，不读 test。错误或暂停阻断后续阶段，不自动重试。结果写入各 run 的 `evaluation_no_mib_dev500/results.json` 和日志末尾，三组结束汇总 `comparison.md/json`。

```powershell
# 首次启动（本轮由助手在准备通过后执行，不要重复运行）
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E135_initial_supervision.run_queue --start

# 后台安全暂停；前台运行时直接 Ctrl+C 也走相同流程
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E135_initial_supervision.control --pause

# 确认 queue_state.json 为 paused 后恢复，可在此窗口 Ctrl+C
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E135_initial_supervision.run_queue --resume
```

训练在已接受批次边界保存 model/Adam/RNG/排列/游标/调度/投影与累计预算；恢复逐位等价，预算不清零。评价在逐例原子提交后暂停，从已提交连续前缀续跑。Ctrl+C 后等待安全退出，不要强杀。STOP/QUEUE_STOP 不自动删除。

状态见 `queue_state.json`；训练心跳见 `runs/e119_A_seed0/heartbeat.json`（B/C 同理）；每阶段日志见 `launch_logs/`；后台总日志见 `background/`。源码和配置在启动前由 `preflight.json` 封存，每次训练保存快照，冻结后不要编辑本实验源码或 README。
