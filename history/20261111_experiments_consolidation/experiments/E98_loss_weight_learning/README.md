# E98：已停止的调权方向与受保护基线

2026-11-07完成用户新授权的五例逐层只读诊断：增强best71的初始及12层共13次共享head读出已保存；没有训练、搜索或测试，所有旧STOP保持。[观察报告](runs/20261106_layer_diagnosis5/report.md) · [逐层查看](../../visualization/E98_layer_diagnosis/20261106_best71_train5/index.html)。

2026-11-05状态核对：用户已停止并放弃旧调权搜索，原STOP保留，不采用搜索权重/单步选优结论，不自动恢复。原固定Loss best58及泛化增强best71仍是有用的来源模型，保留不代表恢复搜索。

- [原固定基底的完整dev500](runs/20261102_fixed_rich96_seed0/evaluation_full_dev500/report.md)：加权Q1.028555439。
- [增强best71固定深度dev500](runs/20261104_generalization96_seed0/evaluation_depth_dev500/report.md)：加权/平均/混合Q1.043560018/1.050714113/1.047137066，500/500最终合法。
- 两者后处理口径不同，不能将Q差直接归因于增强。
- [多轮STOP核验](runs/20261104_dynamic_mixedq_seed0/stop_verification_20261104.json) · [train500搜索STOP核验](runs/20261104_dynamic_train500_v3_seed0/stop_verification_20261104.json)。
- 后续固定目标研究在独立[E103](../E103_offline_loss_alignment/README.md)；[接手总结](../../docs/research_history.md)说明证据边界。

下文保留原基底设计与历史命令，不是当前启动指令；其中预先恢复验证等旧检查不构成现行必测要求。

# 历史：96×12固定丰富Loss从头训练基底

2026-11-02用户明确要求从头重训并直接后台启动。本次只做一个固定系数基底，不继承E56/E94权重，不启动调权、修正器、后处理评价或自动唤醒。原训练分片、索引和权重全部保护；测试封存。

模型使用原矩形几何布局器96×12、1,471,539参数，全部随机初始化和训练。端口网络输入仍为每模块总权重及加权中心，完整连接仅用于Loss。原始local路径、解码、Dropout.05和FP32保持；每模块Adam更新0.5%保护保留，数值失稳则保存失败证据退出，不自动修改结构。

固定十项：位置1、形状.25；穿透/模块负荷均值/最大超10%负荷各.08975162861857286；线长/紧凑度各.16684160541676145；边界/分组/MIB贡献各.6673664216670458。软约束的三项保留原共享公开分母，不改成三种独立归一化。这里展开E90/E94原固定F目标，明确每项系数，未根据新dev扫描权重。

数据为原v5 train16k，固定train/dev各500，公开MIB平均面积派生任务；复用校验过的v6前缀缓存但逐项核对v5成员。batch32、seed0、AdamW decay.0003、clip1、LR1e-3→2.5e-4→1e-4→5e-5。第1–14轮调度保护，第15轮建立平台基准，连续3轮无明显改善降档，完成所有降档且最低档至少2轮才允许早停。best按固定dev500同一Loss，训练500单独报告平台；开发早停不等价于训练目标收敛。4小时或200轮硬上限，先到则保存，不称收敛。

训练入口：`python -u -B -m experiments.E98_loss_weight_learning.train`；`--resume`仅恢复未结束且身份未变化的运行，保留Adam/RNG/排列游标/调度和累计预算。创建`runs/STOP`安全停止，不自动继续下一阶段。后台入口为`launch.ps1`。

输出：`runs/20261102_fixed_rich96_seed0/`，状态见`heartbeat.json`、`progress.json`、`results.json`；`checkpoints/best.pt`和`latest.pt`分别保存。预检只使用登记五个训练成员，验证实际更新、CUDA图、精确恢复、Loss展开、padding、输入边界和数值拒绝；不运行求解器。

具体配置见`plan.json`、`loss_config.py`，复制来源及哈希见`provenance.json`。后续系数学习必须另行登记训练集反馈、固定求解器与匹配继续训练对照，当前仅建立基底。
