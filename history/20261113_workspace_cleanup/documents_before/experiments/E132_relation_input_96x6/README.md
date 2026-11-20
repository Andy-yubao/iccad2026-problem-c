# E132：完整关系输入编码96×6

2026-11-12用户要求暂停E131、将full/pure两组都改为96×6，准备好后后台启动新实验。已暂停的E131原件与完整latest保留、STOP/QUEUE_STOP保持；本实验从头scratch seed0，不继承其权重或Adam。

| arm | 公共结构 | 每层关系通路 | 参数 |
|---|---|---|---:|
| full | 400→192→96入口；96×6/4头/单头24维/SwiGLU192；全局均值拼接192→96→3标准单读出 | 关系偏置＋W/G隐藏聚合 | 786195 |
| pure | 相同 | 无，块仅接收h/valid | 674979 |

完整400维公开描述仍为16原特征＋128自身地址＋128连接行＋128同组成员行。入口仅编码一次，任意地址重命名不变性不保证；训练按原独立随机流逐样本逐轮重命名，监测/推理身份地址。无几何反馈、辅助读出、MIB、固定通道冻结或规格覆盖。

两组共享全部公共初始化张量；额外分支独立CPU流初始化，不改变公共CPU/CUDA RNG。与E131同时变化宽度、深度及相应MLP宽度，不作单因素深度归因；full/pure仍检验额外关系通路的联合价值，参数数不相等。

保持E131原16k/固定train500与dev500、原兼容CPU缓存、独立末端新Loss、FP32batch32、轴交换/Dropout/地址增强、AdamW/原数值与相对更新门禁、四档LR与耐心及每组4h或200轮。反传/监测/best/LR/早停使用同一固定末端目标。原五训练成员必要CPU/CUDA检查与各一次可丢弃受保护更新通过后，重新scratch后台full→pure；错误/用户停止阻断后续，不自动重试。

本次没有后处理、Q、test、绘图或额外候选；用户要求后台启动后不持续监听。启动登记和控制器位于output/P0_scaling96x6_20261112，主状态queue_state.json，日志launch_logs/20261112_P0_seed0；正式run仅保存best/latest和Loss日志后停止。

准备检查命令：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E132_relation_input_96x6.inspect
```

训练入口（准备门禁未释放前拒绝；已有用户在检查通过后启动的授权）：

```powershell
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E132_relation_input_96x6.train --arm full
.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E132_relation_input_96x6.train --arm pure
```

所有算法在本实验独立维护，只共享datasets元数据/缓存。其他旧STOP、E119研究和E101交付保持不变。
