# E7：降低修复难度的布局草稿（已终止）

2026-10-17，用户明确终止本阶段。**不采用本轮新增occupancy损失，不继续矩形感知架构或未完成组合，不补dev100/test50，不自动恢复训练。** 既有geometry目标和E6冻结模型/后处理继续作为已有参照；本实验代码和权重只保留研究证据，不自动替换主线。

完整结果、停止理由、适用范围与证据限制集中在[阶段结论](conclusion.md)。本轮新损失改善草稿重叠和内部空白，但最终Q与控制基本持平，未见平均耗时收益；单独换架构更差。原定50例Q目标没有完成验证，测试数据未读取。组合组在训练初期中断，未评价，不把它记为“收敛失败”。

## 阅读入口

2026-10-17结题后，用户另行授权原attention_geometry best进行100例本地测试并绘制其中10例。该[独立评价入口](runs/20261017_attention_geometry_test100_seed0/README.md)已完成元数据与绘图检查，**测试推理尚未运行**；不恢复训练，原结题事实保留。

- [阶段结论](conclusion.md)：研究问题、实际训练、三组同10例结果、草稿诊断、局限与终止决定。
- [历史设计](design.md)：模型、Loss、调度器、面板和历史命令。页面已标为终止，不构成继续执行授权。
- [结题核验记录](runs/20261017_e7_closeout/results.json)、[结题配置](runs/20261017_e7_closeout/config.json)、[文档检查](runs/20261017_e7_closeout/documentation_check.json)：停止状态、各组事实、累计计时、55份必要证据哈希及文档核对。
- [复制来源与哈希](inherited_sources.json)、[冻结面板元数据](panels.json)：仅共享datasets/scoring，算法副本在本实验独立维护。

## 已完成与取消的工作

| 模型 | 实际状态 | 证据入口 |
|---|---|---|
| attention_geometry | 原训练策略；用户在86.0544轮停止，保留77轮best；同10例Q评价完成，完整dev2500未完成且不补跑 | [训练结果](runs/20261017_factorial_seed0/attention_geometry/results.json)、[10例分析](runs/20261017_attention_geometry_dev10_seed0/analysis.md) |
| attention_occupancy | fast策略；51轮早停，保留49轮best；完整dev2500及同10例Q评价完成 | [训练结果](runs/20261017_remaining_fast_seed0/attention_occupancy/results.json)、[10例分析](runs/20261017_attention_occupancy_dev10_seed0/analysis.md) |
| rectangle_geometry | fast策略；46轮早停，保留42轮best；完整dev2500及同10例Q评价完成 | [训练结果](runs/20261017_remaining_fast_seed0/rectangle_geometry/results.json)、[10例分析](runs/20261017_rectangle_geometry_dev10_seed0/analysis.md) |
| rectangle_occupancy | 第1.3648轮、step853被用户中断；保留初期best/latest，取消后续训练与评价 | [原停止记录](runs/20261017_remaining_fast_seed0/rectangle_occupancy/results.json) |

三组Q检查的样本均事先固定为dev_quality的worker0/11/22/33/44/55/66/77/88/99各1例；后处理和评分相同，测试未读取。原策略与fast策略不同，且没有完整2×2结果，不能把本轮当成严格同训练条件的因果消融。

## 保留的实现与运行证据

模型与训练：[model.py](model.py)、[losses.py](losses.py)、[occupancy.py](occupancy.py)、[train.py](train.py)。输入和解码：[data.py](data.py)、[batches.py](batches.py)、[inference.py](inference.py)。冻结后处理入口：[solver.py](solver.py)；评价及保存坐标绘图：[evaluate.py](evaluate.py)、[render.py](render.py)。保留这些文件仅供复核，不再执行本阶段计划。

必要实现检查也保留，不作为质量证据：

- [五例数值、标签隔离和恢复检查](runs/20261017_preflight_seed0/results.json)
- [完整循环短检查及Windows保存问题说明](runs/20261017_loop_smoke_seed0_r2/analysis.md)
- [fast规则和日志格式检查](runs/20261017_fast_schedule_checks_seed0/analysis.md)
- [单例推理、后处理、评分及高精度计时检查](runs/20261017_evaluation_smoke_seed0_r2/analysis.md)
- [绘图格式检查](runs/20261017_evaluation_smoke_seed0/index.html)

每个运行的config、source_snapshot、日志和唯一坐标记录保持原样，checkpoint不复制、不入Git，不删除数据或环境。结题只增加文档与[STOP标记](runs/20261017_remaining_fast_seed0/STOP)；原user_stop/validation_early_stop事实不改写。新授权前不移除停止标记，也不因历史恢复命令仍可见而续跑。新的研究方向等待用户另行指定。
