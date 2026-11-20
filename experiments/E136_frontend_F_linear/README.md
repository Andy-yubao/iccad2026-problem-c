# E136 D：S0 使用 F，辅助权重线性增加

用户于 2026-11-14 授权准备通过后直接后台启动，沿用 E135 的自动后处理与安全续训。

| 状态 | Loss | 权重 |
|---|---|---:|
| S0 | E135 C 的四项 F | 1/30 ≈ .033333 |
| S3 | 完整 public-priority-v1 | 2/45 ≈ .044444 |
| S6 | 完整 public-priority-v1 | 1/18 ≈ .055556 |
| S9 | 完整 public-priority-v1 | 1/15 ≈ .066667 |
| S12 | 完整 public-priority-v1 | .8 |

辅助总权重 .2，S0/S3/S6/S9 为 3:4:5:6，S9 是 S0 两倍。所有权重全程固定。

`F = 中心容差 + .05*自由形状 + F/P尺寸 + P位置`，严格沿用 C：非 P 模块中心每轴允许 `.25*sqrt(面积)` 偏差，归一化超出部分用 beta=.1 的 Huber；自由形状为非 F/P 的原始 log 宽高比参考平方差；公开 F/P 尺寸和 P 中心项系数各 1，子集独立取均值、空子集为零。S0 不含碰撞/紧凑/线长/G/B 项，后续监督层不变。

E119 结构、S0–S12 读出、712275 参数、36k固定训练集、train/dev500、scratch seed0、FP32/batch32、增强、AdamW/LR/早停/数值门禁均沿用 E135。每组上限 4 小时或 200 轮；从相同随机初始化重新训练，不从 C 的权重续训。全模块预测、不覆盖 F/P、不使用 MIB。选模、反传、监测、LR、早停使用相同 own-Loss。

队列：**D训练 → 冻结唯一 own-Loss-best → 自动原 dev500 后处理 → 汇总与已存 A/B/C、E134 每三层结果的对比**。后处理使用原 `E98.depth_budget.v1`、原面积/规格与独立硬检查；不按 Q 重选权重，不读 test。评价结束将结果同时写入日志末尾与 `results.json`；最终对比在本目录 `comparison.md/json`。错误阻断队列，不自动重试。

准备仅用原五个缓存训练例检验组合目标、初始化和一次受保护更新。续训/队列/评价使用 E135 已验证实现；核对其原证明与相关源码在身份替换后等价，本轮不重复中断模拟或提前跑正式后处理。

```powershell
# 本轮由助手准备通过后后台执行；不要重复启动
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E136_frontend_F_linear.run_queue --start

# 后台安全暂停；前台运行时也支持 Ctrl+C
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E136_frontend_F_linear.control --pause

# 等 queue_state.json 为 paused 后恢复，可在此窗口 Ctrl+C
& .\.conda\envs\iccad2026\python.exe -u -B -m experiments.E136_frontend_F_linear.run_queue --resume
```

续训保存并恢复 model/Adam/RNG/排列/游标/调度/投影及累计预算；预算不重置。评价按已原子提交的连续例前缀续跑。STOP/QUEUE_STOP 不自动删除。

实时状态：`queue_state.json`、`runs/e119_D_seed0/heartbeat.json`；训练/评价日志：`launch_logs/`；隐藏后台记录：`background/`。启动前封存源码与本 README，后续状态写入运行记录，不修改冻结文件。
