# E141：阻隔惩罚、逐层监督及组合（三组）

2026-11-16 用户明确授权：准备三组 batch64 实验，准备完成后台依次训练，每组完成自动验证原固定 dev500。用户确认采用以下口径，不延续 E140 的 G/B×4。

|实验|arm|监督层|新阻隔项|
|---|---|---|---|
|一|e119_blockage|S3/S6/S9/S12 完整监督|有|
|二|e119_layerwise|S1–S12；S3/S6/S9/S12 完整，其他去全部 overlap 项|无|
|三|e119_combined|同实验二|有|

S0 不监督。每组 S12 权重0.8；实验一 S3/S6/S9 均分0.2，二/三前11层均分0.2。原 grouping/boundary 系数不变，新阻隔项系数固定1，与原软项共享原2B及约束数量归一。详见 DESIGN.md。

原 E119 架构712275参数、scratch seed0、冻结36k、原 train500/dev500、原公开面积/无MIB、全部模块预测、FP32 deterministic、AdamW及LR/早停/门禁保留。batch64；每轮末32成员采用同目标/同Adam/同投影的 eager FP32，完整遍历，无丢弃或重复。每组4h/200epoch累计上限。仅使用兼容现有CPU缓存；不读test。

每组统一以自己的固定训练目标反传、监测、调度和选唯一 own-Loss-best；训练成功后先冻结 best，再运行原 dev500 + E98.depth_budget.v1 后处理及独立硬检查，最终汇总 Q/软约束/耗时。原共同末层 Loss 另列；不同 own-Loss 不跨组排名。历史 E134 为batch32，只作背景，不作为严格batch64对照；单seed和自适应早停限制因果推断。

启动：`python -u -B -m experiments.E141_path_supervision.run_queue --start`。本轮用户已授权后台执行。队列顺序为一训练→一验证→二训练→二验证→三训练→三验证；错误或停止阻断后续，不自动重试。状态 `queue_state.json`；训练状态和权重位于 `runs/<arm>_seed0/`；日志 `launch_logs/`；最终 `comparison.md/json`。

停止：在本实验目录创建 `STOP` 或 `QUEUE_STOP`，保留当前安全更新/评价边界；暂停后只有明确恢复才执行 `--resume`，旧实验STOP不动。实验源码、计划和文档启动前冻结，准备检查模型全部丢弃，正式训练重新初始化。
