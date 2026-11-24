# E165 三层初稿与后段联合微调

使用原面积/no-MIB 完整 train36k、原固定 train500/dev500、seed 0 和 batch 64/尾批 32。复用原 CPU 静态与公开角色缓存。训练采用确定性 FP32、AdamW 与 E161 的数值门禁和逐模块位移保护。

性能修复后，每个监督出口的接触项与阻挡项共享同一棵 MST。CUDA Graph 重放前向、反传和位移保护。最后 32 个真实样本复用同一张图，填充位零权重且不计入训练游标。后段解冻时重新捕获联合训练图。原暂停 checkpoint 与迁移记录见 `migrations/eager_to_graph/`。

1. 从零训练宽 64 × 3 的秩 8 静态关系 L0，共 162,815 参数。固定 dev500 按 own-Loss 选 best，保存原 checkpoint，再复制到 `backups/l0_pretrained_best.pt`，核对 SHA-256 后才启动后段。
2. 从零初始化原 E151 题目编码器和 S0/S0/S1/S1/S2/S2，冻结 L0 整整 20 轮。第 21 轮解冻 L0，后段 AdamW 动量继续沿用，L0 使用后段当时学习率的 0.1 倍，此后同步降档。后段训练期 L0 dropout 始终关闭。只从已完整训练的联合轮次按固定 dev500 own-Loss 选终末 best。总模型 736,217 参数。
3. 自动以终末 best 运行原 no-MIB dev2000 和原 E98 固定后处理预算，每例独立硬检查。报告混合 Q、软约束、完整耗时及两组 dev 子面板，不按 Q 选权重。

L0 保留 0.25 倍重叠。后段目标是 `0.05*L0 + 0.1*L4 + 0.1*L8 + 0.8*L12`，L4 去直接重叠，L8/L12 保留完整重叠。四个出口均保留原接触项并加入 E161 的 0.25 倍 MST 阻挡项。L0 单独阶段最多 4 小时，后段合计最多 8 小时，每个训练进程最多 200 轮。发生错误、停止，或没有完成任何联合轮次时，队列不会用其他权重代跑 dev2000。

## 日志

| 内容 | 路径 |
|---|---|
| 队列 stdout / stderr | `background/queue.stdout.log` / `background/queue.stderr.log` |
| 续训队列 stdout / stderr | `background/queue.resume.stdout.log` / `background/queue.resume.stderr.log` |
| L0 训练 | `background/e165_l0.log` |
| 后段冻结与联合训练 | `background/e165_backend.log`，以 `[UNFREEZE]` 分界 |
| 逐批性能与训练状态 | 对应 `runs/e165_l0_seed0/` 和 `runs/e165_backend_seed0/` 下的 `steps.jsonl`、`progress.json`、`heartbeat.json` |
| dev2000 进度与最终摘要 | `background/dev2000.log`、`evaluation/background/quality.log` |
| 队列与终末结果 | `queue_state.json`、`evaluation/output/state.json`、`evaluation/output/results.json` |
