# 机制消融与核心深度确认

**2026-10-23用户已收回后续启动授权：只保留当时正在运行的E39 no_gap训练继续。定时跟进e38已PAUSED，尚未启动的四组seed1/2训练取消。CANCEL_PENDING阻止本入口重新启动；queue_cancelled.json记录当前训练与队列隔离，queue_guard.ps1仅等待当前训练自然结束后清理已禁用的协调器，不启动新训练或评价。下文保留原计划，不再作为启动授权。**

2026-10-23，在E39/E40首轮完整结果结束后登记并启动。[入口](runner.py)、[启动记录](launch.json)及[控制/源码身份核对](control_reuse.json)保留。用户本轮明确允许后台长训练、继续有依据的模型优化；Loss、v5与测试封存边界保持不变。

执行顺序：E39 rectangle_no_gap seed0 → E38普通4层seed1 → E38普通12层seed1 → 普通4层seed2 → 普通12层seed2。单GPU依次执行，每组冻结唯一best后完成完整dev及一次终末500，再进入下一组；CPU总计四线程，与GPU训练分时。E39消融检验碰撞收益是否与边界/MIB退化共同依赖间隙/穿透特征；普通深度确认检验seed0核心收益能否跨初始化重现。E40本实现完整dev代理基本不变、碰撞与约束严重度退化且推理更慢，停止额外消融，不为凑齐矩阵扩大无信号路线。

所有新运行从头初始化，batch32、相同4档LR/早停，每组累计最多4小时/200轮，保存best/latest和完整恢复状态。五组训练合计保护上限20小时，按此前吞吐预计约5小时，估计非承诺；验证/终末另计。种子确认复用同一份E38冻结源码，seed0四层与12层成果只读复用，不重复重训。每个新种子必须做同种子4层/12层配对，不把模型间不同随机种子当配对。

E39输出：`experiments/E39_rectangular_attention/runs/20261023_rectangle_no_gap_w96d12_rectangle_no_gap_seed0/`。

E38输出：`experiments/E38_model_architecture/runs/20261023_depth_confirmation_<arm>_seed<seed>/`，arm为base96x4或deep96x12，seed为1/2。控制与候选采用同一数据、监测和停止规则，各自唯一best的终末仅作结果，不反选checkpoint。

命令与逐阶段状态由`active.json`、`progress.json`及`console.log`记录；各运行config保存实际命令和源码快照。后台启动器保留exec会话，当前ID见launch.json。跟进目录STOP请求当前子阶段安全停止；移除已核实的跟进/子阶段STOP后，用独立环境运行`runner.py --resume`，保留累计预算，不覆盖已完成阶段。不修复运行中源码；失败先保存证据再诊断。E39/E40/E38源码此阶段保持冻结。

完成后报告三种子逐种子与汇总验证代理、完整dev、draft冲突/约束及推理成本；终末配对效应同时披露案例抽样、分片相关与限时求解的不确定性。三种子只能提供初步稳健性，不代表充分的种子分布统计。如果结构消融确定了更优候选，再做其与同主干的必要种子确认，不按单次终末Q决定。研究尚未结题，持续跟进保持开启。
