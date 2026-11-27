# 当前实验区

2026-11-12新增：[E133_flow_matching](E133_flow_matching/README.md)为用户授权的单组Flow Matching首轮实验，96×6条件速度场/920643参数/冻结16k/纯FM Loss，准备通过后独立后台启动；总8h包含训练和固定dev500终末评价，唯一FM-Loss-best冻结后用Euler8及原E98深度后处理。主状态E133/queue_state.json；不读test、不新增图/候选，错误或停止不自动推进。旧实验与研究/交付身份保持。

2026-11-12最新：[E132_relation_input_96x6](E132_relation_input_96x6/README.md)按用户要求将full/pure两组升级96×6，原五例检查通过后后台从头训练；不后处理。E13164×4已用户暂停，完整状态和STOP保留，pure未启动，不自动恢复。

2026-11-12新增用户授权的准备入口：[E131_relation_input](E131_relation_input/README.md)，P0完整关系输入编码，full机制组vs纯Attention组；仅准备、训练禁用、STOP保留，本次无后处理。原五训练例CPU/CUDA图检查通过，零优化器更新。以下2026-11-11整理为历史完成记录，不妨碍本次明确新增入口。

2026-11-11整理后，`experiments/` 仅保留两个研究身份；其余全部实验原件（E98、E101–E118、E121–E130，共29个目录、145,392个文件）已同盘归档至 [history/20261111_experiments_consolidation](../history/20261111_experiments_consolidation/README.md)，逐文件清单与 SHA256 核对见该归档 manifest。

- [E119_global_initial_generator](E119_global_initial_generator/README.md)：global2 初始生成器，当前主要研究基线（`protocol/research_baseline_v3.json`）。训练与原固定 dev500 后处理已完成（queue_state complete，末层 dev Loss 0.333128，混合Q 1.021568，500/500硬合法）。
- [E120_node12_collision](E120_node12_collision/README.md)：Node12-Collision 结构对照（混合Q 1.039004）。训练与原固定 dev500 后处理已完成（queue_state complete）。

两者均为独立本地算法副本，复用 `datasets/static_tensor_cache_v1` 共享缓存；test 继续封存。归档实验的恢复须先获用户明确授权并登记；旧命令只追溯。

[带日期状态](../docs/current_status.md) · [全部实验实际目录](../docs/experiment_catalog.md) · [研究综合](../docs/research_history.md)。
