# E64：E42统一固定Loss从头重训

本轮已完成：43轮正常早停，best40，训练约32分钟。固定dev500 G0.368244（原E42为0.357200），完整dev2000 G0.365703，真实交面积均值7.146%（原6.302%）；flash500全硬合法、加权Q1.030026，尾部改善但整体未升级。原E42权重保持冻结，测试未读。见[完成复核与曲线](runs/review.md)、[100例无评分图集](runs/20261027_e42_fixedG_scratch_seed0/draft_comparison100/index.html)。E64保留为统一协议对照，第二阶段尚未启动。

2026-10-27用户明确授权：先建立训练与选模统一的E42基线，准备通过后后台启动。本轮只做第一件事；后续表达能力及泛化研究尚未启动。原E42／flash、E50／pro与所有旧checkpoint保持冻结。当前状态见[state.json](state.json)，执行证据在[runs/](runs/)。

冻结 **E42原实际反传的八项完整G**，从第一个batch起固定：`center_local + .25 shape + .25 relative_local + .5 overlap + .5 contact + .25 boundary + .1 mib + .1 bbox`。定义见[loss_config.py](loss_config.py)、[losses.py](losses.py)，历史解释见[原E42 Loss](../E42_joint_regularization/loss_definition.md)。不采用E63中心0.75或softband，也不把occupancy项并入新Loss。训练、每例及跨例聚合、train/dev监测和完整开发评价调用同一函数；未来获授权测试也须使用此定义。best、LR、早停只读固定dev500同一G。这个冻结定义作为下一阶段架构研究的参照，除非用户明确修订。

## 固定配置与解释边界

- 从随机初始化重训，seed0；原E42 96×12 rectangle模型、公开输入、FFN dropout=.05、AdamW decay=3e-4。模型／features／batches字节保持原样，不加图聚合前归一化或E61步长修复。
- 原v5 train16000逻辑顺序、每轮`default_rng(seed+epoch)`排列；固定train_monitor500／dev_monitor500，不采用新增64k。共享缓存只读复用，不读取测试。
- batch32、FP32、clip1、四档LR `1e-3→2.5e-4→1e-4→5e-5`；15轮起参与调度，连续3轮无明显改善下降；三次下降后最低档至少完整训练与验证两轮再允许早停。显著改善门槛为`max(1e-5,.001*基准)`，best保存门槛为绝对改善>1e-5，二者均只使用G。
- 原E42前两轮渐入取消，G+occupancy选模取消。执行采用已核验CUDA graph与GPU常驻输入；保持FP32／原AdamW语义，但不宣称与历史eager训练轨迹逐位相同。最多4小时／200完整轮，预算停止不称收敛；预计约40–100分钟，未经本轮全量测量。
- E57已有相近的16k／G从头训练对照，其训练行排列和运行目的不同；本轮是建立原E42顺序的清晰基准，不是首次证明统一G可以训练。单seed不能把差异全归因于某一规则，也不能证明Loss已最优或表达能力不足。用户草稿评价优先，G为主要数值参考，Q为辅助。

精确授权、配置、停止条件见[plan.json](plan.json)；复制来源和旧权重保护见[provenance.json](provenance.json)。不跨实验导入私有实现；数据处理、评分、布局绘图分别复用datasets／scoring／visualization。

## 执行、停止与恢复

在项目根目录，独立环境为`.conda/envs/iccad2026`。五例预检查执行：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E64_e42_unified_retrain.preflight
```

预检查只读取五个train成员，覆盖原E42 G／梯度一致、padding、真实Adam更新、Dropout下中断恢复逐位一致、四档调度和旧权重保护。证据：[预检查结果](runs/20261027_preflight/results.json)。启动器要求当前源码哈希与通过的预检查一致。

```powershell
& '.\experiments\E64_e42_unified_retrain\launch.ps1'
```

后台入口仅运行一次训练，再为固定G所选唯一best完成raw dev2000、原flash dev500、同100例无评分三栏图和学习曲线。CPU评价用2个进程各2线程，图片用4个进程各1线程，总线程预算4；不自动打开评分界面。结束证据为[runs/review.md](runs/review.md)与[runs/review.json](runs/review.json)，不自动升级基线或追加训练。

训练产物位于`runs/20261027_e42_fixedG_scratch_seed0/`，包含配置、源码快照、固定面板曲线、best与可恢复latest。日志：[train.log](runs/train.log)；总状态：[state.json](state.json)；启动记录：[launch.json](runs/launch.json)。GPU常驻输入约数GiB，使用单张本机RTX 5080 Laptop，实际用量写入日志。

安全停止：创建`experiments/E64_e42_unified_retrain/runs/STOP`，当前检查边界保存后退出且不执行下一阶段。恢复先移除自己创建的STOP，再执行相同启动器（内部`--resume`）；恢复保留优化器、LR档位、随机状态、样本游标、已见成员和累计时间。硬中断最多回退约30秒到最近latest；预算已经结束的运行不自动延长或重训。失败或非有限值保存latest并停止，不自动变更架构、精度、batch或Loss。
