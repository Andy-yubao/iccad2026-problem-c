# E91正式加速与追加1小时续训

2026-10-31，用户要求“优化一下训练速度并继续训练”，并确认“追加1小时”。已将此前隔离验证的步长保护CUDA图接入正式路径，后台执行`pipeline --resume`，控制器PID40620。

## 修改范围

新增实验内`captured_projection.py`，按原样录制GraphStepControl的before/after算子，降低Python逐参数组提交开销。AdamW仍在图外，实时LR、梯度裁剪1、每组0.5%相对位移规则、数值保护、Dropout、FP32和batch32不变；不改变模型、公开特征、32k数据、采样次序、Loss公式/系数/归一化、监测成员或选模规则。

恢复时额外逐项恢复last_scales，以保持完整统计状态；旧steps与clipped counters继续保留。捕获检查确认不改参数、计数或CUDA随机状态。

入口从计划读取本次1小时窗口，旧费用作为固定offset。原32轮训练已用11071.06987560005秒（含旧显存事故费用），本次新给3600秒，总阶段上限14671.06987560005秒，累计18小时上限保持。载入/恢复时间仍计入窗口；没有清零旧账或重置调度。

## 等价性检查

同原五个真实训练成员1683/10027/5474/18140/25001，未访问测试：

- 集成后的原有五例预检通过，覆盖固定Loss、公开输入、120节点padding、所有图层更新、0.5%边界、普通/捕获更新等价、非有限值拒绝和真实Dropout/Adam恢复。[日志](preflight.log)
- 使用已训练的latest32443、当前LR0.00025和真实Adam矩，与保留的旧cuda_step实现分别连续执行6步；模型、Adam、随机状态、scales和全部clipping counters逐位相同。
- 新路径先2步，再保存/重建/恢复跑4步，结果与新路径连续6步逐位相同。[训练状态检查](trained_resume_checks.json)、[脚本](check_trained_resume.py)、[日志](trained_resume_checks.log)
- 此次空闲GPU下的同五例微基准，原路径185.75ms/步，新路径103.10ms/步，约1.80倍。仍不是正式整轮吞吐，实际收益以续训日志的同轮100步区间为准。

## 正式断点迁移

从latest的**epoch32、step32443、cursor14176**接续第33轮，不从best32000回退。学习率保持0.00025，Adam矩/步数、CPU/CUDA/Python/NumPy RNG、调度器、历史、当轮排列、seen、游标、投影计数与scales均逐项核实未变。

best/latest仅重绑源码身份和授权预算；latest的budget-finished标记因用户授权续训清除。原checkpoint、结果、费用和源码均保存在[before](before/)，历史训练记录不追溯改写。追加窗口不重开保护期或清空停滞计数。

[预算授权](authorization.json) · [迁移核查](migration.json) · [迁移脚本](migrate.py) · [迁移后源码](source_after.zip) · [后台启动](launch.json)

当前训练日志：[stdout](../training.speedup_resume.stdout.log)、[stderr](../training.speedup_resume.stderr.log)。仅续训本组；不自动Q、全量开发评价、图库或新候选，测试继续封存。

## 正式运行的初期实测

已核实正式latest推进到33300步，LR仍为0.00025。排除重载后的首个跨阶段区间及每轮监测，7个同轮100步区间的中位耗时为110.33ms/步；优化前近期约201.71ms/步，初期实测约1.83倍。一次GPU采样为92%利用率、3184MiB显存。这是初期短窗口测量，不代表整个新增1小时都固定此速度。[运行观察](runtime_observation.json)。
