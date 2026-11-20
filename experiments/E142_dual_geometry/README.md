# E142：模块动态需求与模块对关系共同驱动布局更新

2026-11-16 用户已认可新架构，并明确授权准备通过后直接后台训练、训练结束自动验证原固定dev500。用户已说明暂不考虑续训，本组从头训练scratch seed0。

模块动态几何49维＋原公开静态属性16维，经共享需求编码器生成64维u；每层融合自身隐藏状态与需求后生成QKV。保留原18维模块对attention bias，并加入包含接收者/发送者状态及模块对几何的8维关系消息，在四个头的value通路中聚合。保留自身残差、原两初始图块、12层width64和原共享布局头。不增加局部动作头、局部贪心监督或中间布局层。数学定义与49维字段见 DESIGN.md。

监督与Loss完整复用E139：

|层|目标|权重|
|---|---|---:|
|S2|完整原Loss去全部直接overlap项|.05|
|S6|同上|.05|
|S8|完整原Loss|.05|
|S10|完整原Loss|.05|
|S12|完整原Loss|.8|

其余层不加直接监督。无E141路径阻隔Loss、无G/B×4、无拓扑标签。原reference/public-hard各项系数、聚合保持。新几何摘要仅依赖当前预测和公开输入，并detach；原pair、state_embed与主干反传路径保持。

本轮沿用batch64；36k每轮562个64及一个32，末批采用同目标同Adam/投影的eager FP32，不丢题不重复。895123参数，全部训练；FP32 deterministic/TF32关闭，原增强、AdamW、LR/早停、Adam前数值门禁和.5%模块更新界限。每组累计4h/200epoch上限。复用固定36k和原train500/dev500兼容CPU缓存，不重新抽样或构建静态数据，不读test。全部模块仍预测位置/形状，F/P仅为公开条件，硬规格由原后处理满足。

启动：`python -u -B -m experiments.E142_dual_geometry.run_queue --start`。通过preflight后按本轮授权直接后台执行。队列为train→冻结唯一own-Loss-best→原dev500/E98.depth_budget.v1后处理→独立硬检查→Q报告及历史E139对比。每组验证完成立即打印[Q]摘要到评价及后台队列日志，并写background/quality.log。Q不参与选模。

实时状态：queue_state.json及runs/e142_dual_geometry_seed0/heartbeat.json；日志background/queue.stdout.log和launch_logs；Q原件runs/e142_dual_geometry_seed0/evaluation_no_mib_dev500/results.json。STOP/QUEUE_STOP和错误均阻断后续，无自动重试或追加seed。安全暂停后才显式--resume，恢复全部训练状态与累计预算。

E139历史对照为batch32；新架构和batch均有变化，单seed/自适应早停不能支持纯架构因果结论。历史源码、权重、STOP和结果原件均保留。
