# E63：E42 温和放宽中心约束后训练

2026-10-27 19:35（北京时间）全流程完成：三组分别39／43／41轮正常早停，best为36／40／36轮，总训练约90分钟。完整raw dev2000、每组一次flash dev500、各100张同例无评分图及[汇总报告](runs/review.md)已保存。中心权重0.75较续训对照交面积均值降低2.30%，但中心与相对位置误差略升；小偏移软化重叠收益更弱。未替换原E42，测试封存。旧训练记录中的`full_dev_complete:false`和`final_postprocessed_Q:not run`是训练阶段快照，后续已完成评价以各运行`full_dev.json`及`terminal/summary.json`为准。

用户于2026-10-27明确授权实际运行后训练，允许自行设计温和放宽方式；不做完全去中心监督分支。动机来自[E62同5个训练例的实际Adam更新检查](../E62_e42_loss_update_audit/runs/20261027_best51_five_train/analysis.md)：中心梯度与重叠梯度抵消，并影响全局裁剪；这只支持优先尝试，不能预言完整训练必然改善。

三组均从冻结 E42 best51（epoch51/step25500）相同权重分叉，以原 v5 train16000、固定train/dev监测各500后训练，seed0。

| 组 | 中心项改动 | 保留约束 |
|---|---|---|
| control | 原始中心项，系数1 | 原Loss续训对照 |
| weight075 | 中心系数固定0.75 | 全程保留75%中心监督 |
| softband005 | `0.5ρ(e)+0.5ρ(sign(e)·max(|e|−0.05,0))` | 小偏移保留至少一半原梯度；单轴偏移≥0.15等效边长时梯度恢复原值 |

`e=state_xy/sqrt(area)-target_xy/sqrt(area)`，ρ为原SmoothL1(β=.1)。其他七项、mask、归一化、聚合、96×12模型和公开输入保持一致。无图聚合前输入归一化，无新图步长保护。每组的Loss在反传、监测、完整开发评价及未来获授权测试中保持固定；按自身dev500 Loss选唯一best，包含epoch0初始权重。

三组共同采用**权重续用、Adam和调度器重置**的新后训练阶段，排除历史动量这个混杂因素；不是精确续接旧运行。旧训练预算及checkpoint完整保留。新阶段各最多2小时/200轮，总训练上限6小时，串行使用GPU。batch32、FP32、dropout .05、AdamW decay .0003、clip1、四档LR及第15轮后降档/早停按协议。严格断点恢复保留模型、Adam、RNG、成员游标、调度和累计预算。

固定方案见[plan.json](plan.json)，初始权重身份见[provenance.json](provenance.json)。实现从E56独立复制，模型/features/batches保持E42字节身份；不跨实验导入私有算法。只复用datasets、scoring及visualization。

## 执行与恢复

在项目根目录执行：

```powershell
.conda/envs/iccad2026/python.exe -u -B -m experiments.E63_e42_center_relaxation.preflight
.conda/envs/iccad2026/python.exe -u -B -m experiments.E63_e42_center_relaxation.pipeline
# 仅未完成阶段恢复，已完成阶段不重跑；不得同时启动第二控制器
.conda/envs/iccad2026/python.exe -u -B -m experiments.E63_e42_center_relaxation.pipeline --resume
```

在本实验`runs/STOP`创建文件可安全暂停全流程；恢复前移除此文件。源码/计划身份变动会拒绝续跑，不覆盖旧运行。

## 证据与评价

- [五例短检查](runs/20261027_preflight/results.json)：仅E62原5个训练成员，Loss一致性、梯度、CUDA及实际Dropout/Adam断点恢复。
- [实时状态](state.json)：真实控制器及当前子进程PID、阶段与输出。
- 每组`runs/20261027_<arm>_seed0/`：实际config、源码快照、固定曲线、best/latest及累计预算。
- 完成后[汇总报告](runs/review.md)、[逐例配对](runs/review.json)、[共同口径曲线](runs/learning_curves.png)。

每个自身Loss best只作一次完整raw dev2000和既定flash dev500终末评价，再生成相同100例reference／本轮draft／冻结E42图，无评分控件。使用原始参考坐标绘图，避免反归一化伪重叠。三组总Loss定义不同，不能用总Loss跨组排名；重点看共同原G、真实交面积、穿透、相对位置与形状。Q仅辅助，不重选checkpoint。单种子、自适应开发证据；不自动升级基线，测试封存。
