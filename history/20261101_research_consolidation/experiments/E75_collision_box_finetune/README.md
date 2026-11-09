# E75：E74候选上的真实交面积与外框后训练

2026-10-29已终止：3374步/3轮后触发有限值爆炸保护，latest完整保存，best仍为源模型epoch0，未执行终末Q。图参数虽固定，其他主干/输出参数仍累计漂移；同批次源E74健康、E75失稳。保留原源码与证据，不自动续跑。下一候选为[E76完整冻结主干＋有界修正](../E76_bounded_refinement/README.md)。

2026-10-29，按用户授权继续优化。E74草稿收益一致，flash500加权Q均值1.022511优于E42的1.025860，但配对worker重抽样区间仍微跨零；先保留候选，不宣布稳固升级或更换冻结基线。[E74复核](../E74_collision_32k_posttrain/runs/completion/inspection.md)包含同2000例/100图、弱端口子群、失败案例和限制。

## 先用反例淘汰不合适的组合

最初拟用`G+2C+0.5 relu(log(B/Bref))`，其中C沿用E74的方向门控碰撞代理。五个固定训练例的输出空间短步显示：虽然五例外框缩小，但四例交面积增加。该组合在正式长训练前放弃，源码快照、方案与预检保存在`runs/linear_box_preflight/`和`runs/linear_box_proposal.json`，不算一次正式训练、不重写旧证据。短步不是实际Adam参数更新，也不证明训练必然失败；它揭示了可避免的目标冲突。

正式组合改为直接惩罚真实成对交面积，并使用接近参考时斜率较小的平方单侧外框惩罚，避免线性项一开始主导压缩。五例新方向均降低交面积，但三例外框小幅增加，不能承诺所有目标同时改善，也不依据五例宣称泛化成功。

## 固定目标与继承

`L = G + 2 I + 5 E`，其中`I=sum_{i<j} actual_intersection_ij / public_total_area`，与原始草稿交面积诊断一致；`E=relu(log(B/B_reference))^2`。G八项公式、权重、参考位置/形状/近邻监督完整保留；旧方向门控C仅诊断，不再参与总Loss。

真实交面积在完全包含时可能对中心平移没有梯度，因此保留G的穿透项，不声称I独自解决避碰。E只对极值块有直接梯度，也不保证内部填空或每例参考外框都可行；它是软监督。二者与G仍可能冲突，完整同例几何/Q是必要核对。

从E74由自身固定Loss选出的best42/42000步加载完整192宽权重，不再次扩宽、不从头重学、不用终末Q换checkpoint。原E42和E74 checkpoint只读保护。重新建立Adam/四档调度器作为**新Loss的新后训练阶段**，历史预训练与E74约2小时预算单独记录；不是延长E74或清零它的累计预算。

沿用冻结32k、原train/dev500成员、batch32、seed0、FP32、全程固定图聚合Linear和其他可训练模块0.5%实际Adam联合步长上限。训练、监测及完整dev均使用同一固定新Loss，epoch0可选，best/LR/早停只看新Loss开发均值，无渐入。不同总Loss不直接横比，旧G、交面积、外框/空白、原始合法性和固定flash提供解释。

## 运行

```powershell
.conda/envs/iccad2026/python.exe -u -B -m experiments.E75_collision_box_finetune.preflight
.conda/envs/iccad2026/python.exe -u -B -m experiments.E75_collision_box_finetune.pipeline --resume
```

[最终预检](runs/preflight/results.json)包含同五个真实训练例（两个来自新增16k），原始/缓存一致、E74初始输出完全相同、平方外框单侧值/梯度、无碰撞零代价/连续性、实际步长边界、冻结图参数和真实Dropout/Adam精确恢复均通过。

单组最多2小时/200轮，预计约1.5～2小时；32k常驻输入约5.9GiB，单GPU，CPU终末2进程×2线程。输出`runs/20261029_e74_collision_box_seed0/`。创建`runs/STOP`安全停，移除后同命令`--resume`，身份和预算不清零。训练后唯一own-Loss best完成dev2000、一次flash500和固定100图reference/current/E42无评分展示；与已存E74坐标/评分只读比较，不重复E74求解。

E75是第四个正式候选，长期总账及30分钟跟进仍在[E72 study](../E72_topology_posttrain/study.json)。E74优点与原基线保留，测试封存。
