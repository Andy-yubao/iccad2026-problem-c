# E166 联合后段的软约束监督

复用 E165 独立预训练 L0 best67 的备份，校验 SHA-256。按 E165 的模式从零训练题目编码器和完整 S0/S0/S1/S1/S2/S2 后段。前 20 轮冻结 L0，第 21 轮解冻，L0 学习率为后段当时的 0.1 倍。两个参数组同步降档，后段保留 AdamW 状态。总参数 736,217。使用原完整 train36k、固定 train500/dev500、seed 0、batch 64/尾批 32、确定性 FP32 CUDA Graph。训练上限 8 小时或 200 轮。

沿用 E165 的出口权重和重叠口径。仅 L12 的 reference 系数降为 0.25，固定尺寸与预摆位置监督保持原系数。L8、L12 分别加入系数 0.01、0.02 的分组和贴边代理项，两项均使用归一化缺口的 `d/(d+0.01)`。分组选中 MST 边按原接触距离计罚，贴边按模块选取最差的必贴边。两项都除以原 no-MIB 软约束容量。接触、阻挡与分组代理项共享每个出口的一棵 MST。

固定 dev500 按联合阶段 own-Loss 选权重，队列自动用原后处理预算评价原 no-MIB dev2000，并逐例检查硬合法。软约束结论以实际 grouping、boundary 和归一化软违规率为准。

- 训练日志和心跳见 `background/e166_backend.log`、`runs/e166_backend_seed0/heartbeat.json`。
- 队列见 `queue_state.json`。启动和队列日志在 `background/`。
- dev2000 日志与结果见 `background/dev2000.log`、`evaluation/output/results.json`。
