# ICCAD 2026 Problem C

[E100关系引导布局网络](experiments/E100_relation_guided_placer/README.md)训练与固定dev500数值检验已完成：64轮正常早停，best61；相同完整后处理500/500硬合法、零回退，加权Q1.031556747，未优于E98的1.028555439。草稿成对交面积8.09%对E98的6.97%，精确覆盖≤10%的模块38.59%对48.20%；近邻方向准确率83.78%，坐标启发式83.54%。4层几何更新降低自身初始重叠，但尚无整体升级证据。完整均时227.39ms/P95 521.00ms，166例≥250ms，测试封存；不新增训练或自动候选。[检验报告与图集入口](experiments/E100_relation_guided_placer/runs/20261103_relation_guided_seed0/evaluation_dev500/report.md)。

[E99吸引排斥前置研究](experiments/E99_attraction_repulsion/research/README.md)本轮未达标；用户2026-11-03清除目标后，已暂停自动唤醒并阻止后续阶段，当前无后台求解。完整dev500：500硬合法、Q1.013885087，完整均时216.279ms/P95 536.478ms/最大2311.908ms，135例≥250ms。[固定100例四阶段图](experiments/E99_attraction_repulsion/runs/research/20261103_R23_frozen_R15_complete_dev500/four_stage100/index.html)及[验收复核](experiments/E99_attraction_repulsion/runs/research/20261103_R23_frozen_R15_complete_dev500/acceptance_review.md)已完成。另500训练成员复核：原流程Q1.013305624、AR Q1.015056201，两臂全部硬合法，未找到稳定AR升级。C3/原后处理冻结、旧证据保留、测试封存；新实验等待明确恢复。[状态](experiments/E99_attraction_repulsion/research/study.json)。

[E99统一调参入口](experiments/E99_attraction_repulsion/TUNING.md)已交付：修改同一tuning.json后，一条命令运行固定100例AR＋后处理、原流程对照统计，并按用户要求生成“原始草稿／此前最优AR草稿／当前AR草稿／当前全流程最终布局”四列主图；支持配置快照、缓存复用与显式断点恢复。本组20轮/单步短边5%/累计40%、尺寸因子最多1.5已完成：草稿交面积均值2.3010%→1.6765%，但子集加权Q1.013053→1.013471，未改善加权质量。原v3 Q1.011652及全部旧证据保留，无活动求解、新训练或自动扫描，测试封存。[运行记录](experiments/E99_attraction_repulsion/runs/tuning/index.md)。

[E98固定丰富Loss布局基底](experiments/E98_loss_weight_learning/README.md)及固定dev500完整评价已完成：best58 Q1.028555439、500/500硬合法、零回退，均时.232293秒/P95 .521284秒。用户新授权的[首轮训练集反馈调权](experiments/E98_loss_weight_learning/weight_learning/README.md)已后台启动：同best58起点的原系数对照／对称随机扰动两支各训练4轮，24个训练成员×2次求解决定是否接纳新系数；冻结选择后才做dev终末对照，不回流调权。最多1小时／一个外层回合，测试封存。[调权进度](experiments/E98_loss_weight_learning/runs/20261102_weight_round1_seed0/state.json)。

当前方向：GPU神经网络＋非神经网络混合求解，以更好的网络草稿支持更简单、迭代更少的后处理；质量优先，最终全部硬合法、加权Q<1.01，逐例完整延迟尽量<0.25秒。[E97本轮研究已收尾](experiments/E97_gpu_hybrid_solver/runs/final_report.md)：8个候选全部完成，固定dev500全部硬合法且实测完整耗时满足目标，但质量未达标。同3步预算最低Q1.169365来自候选2，草稿重叠较差；最后候选8 Q1.233626、均时.14017秒/P95 .21869秒，11例低质量回退仍未消失。未升级达标模型，自动唤醒已关闭。

[E97第二研究窗口已收尾](experiments/E97_gpu_hybrid_solver/runs/round2_final_report.md)：用户延长截止至北京时间17:45，随后要求“不再训练，直接审计收尾”。7个正式候选完成并审计，候选8仅预检通过、未训练；保留候选5，完整dev500 Q1.012685893、500例全硬合法、0回退，均时.204464秒/P95 .521080秒，126例超过.25秒。优于当前C3控制，但质量与延迟尾部未达标。无活动训练/评价，自动唤醒已关闭，测试封存。

用户新授权的[旧E85完整后处理单线程补测已完成](experiments/E97_gpu_hybrid_solver/runs/fullrepair_singlethread_findings.md)：10臂各500例全合法、零回退；相同完整配置E85 Q1.015922，C3/C6约1.0133，修正此前弱三步流程下的候选判断。质量和速度仍未同时达标。全部求解已退出，未新增训练、未开放测试、未恢复自动唤醒。

E95 直接外框 8.46 候选已确认失败，完整证据保留。E96 完全压缩、外框权重0.5的训练与评价已完成：单次草稿交面积1.0854%、平均N5=8.342，flash加权Q1.026523，未升级冻结E90；本轮在9872步上限结束。[实验入口](experiments/E96_port_compression/README.md) · [实时状态](experiments/E96_port_compression/state.json)。

## 保留的基线与当前实验

| 目录 | 保留用途 | 已保存同dev500结果 |
|---|---|---|
| [E42](experiments/E42_joint_regularization/README.md) | 冻结draft＋flash | Q 1.025860 |
| [E50](experiments/E50_anytime_quality/README.md) | 质量优先pro，约5秒预算 | Q 1.012043 |
| [E74](experiments/E74_collision_32k_posttrain/README.md) | 较强布局草稿和尾部参照 | flash Q 1.022511 |
| [E90](experiments/E90_auxiliary_weight_placer/README.md) | 当前修正器的冻结L来源 | flash Q 1.022348 |
| [E81](experiments/E81_multisource_refinement/README.md) | 可独立调用的旧神经修正器 | flash Q 1.016242 |
| [E85](experiments/E85_cooperative_space_refinement/README.md) | 神经修正＋确定性几何算子对照 | cooperative flash Q 1.014672 |
| [E95](experiments/E95_single_pass_corrector/README.md) | 已失败的直接外框8.46候选 | flash Q 1.251831；早期归一化版本 Q 1.031633 |
| [E96](experiments/E96_port_compression/README.md) | 当前独立F：完全压缩，外框权重0.5 | 已完成；flash Q 1.026523，N5目标未达 |

这些是自适应开发结果、不同预算与流程的参照，不是统一排名、正式比赛分数或独立泛化结论。冻结E42/E50不因较低点估计自动升级。

## 工作入口

- [执行协议](docs/protocol.md) · [工作区结构](docs/workspace_structure.md) · [GPU训练工程](docs/gpu_training.md)
- [数据版本与位置](docs/datasets.md) · [Git约定](docs/git.md) · [实验索引](experiments/README.md)
- [官方题目](references/official/problem_statement.md) · [研究导读](references/research_guide.zh-CN.md)
- [本次整理与迁移清单](history/20261101_research_consolidation/README.md)
- [2026-10-30项目全貌报告](output/2026-10-30_项目研究全貌报告/README.md)（历史快照，不代表当前执行状态）

独立环境`.conda/envs/iccad2026`；数据路径由protocol/paths.json解析。测试封存，无评分界面；本轮12小时/最多8正式候选预算因8候选用满收尾，自动唤醒已关闭，账本见E97。实验执行状态与精确数字以所属runs记录为准。
