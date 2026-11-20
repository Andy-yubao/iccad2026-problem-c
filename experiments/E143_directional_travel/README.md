# E143：受限穿透增量的四方向可移动距离

2026-11-16用户明确授权：阈值使用0.1，其他先不改；准备检查通过后后台启动训练。延续本轮自动原固定dev500验证。

以E142 best55既有结果为对照，仅调整模块动态反馈：

- 保留四方向0.25sqrt(A_i)刚性试移响应，计算方式逐值不变。
- 删除四方向1.0sqrt(A_i)响应。
- 原escape4替换为travel4：保持形状和其他模块不动，沿当前方向连续移动，使所有模块对相对穿透深度不超过起点值＋0.1；在第一个越阈区间入口停止。
- 每方向再取两个上限：到当前布局对应边界的距离；当前对应边长（左右宽、上下高）。边界在起点冻结。返回距离除sqrt(A_i)，保留原asinh压缩。代码方向顺序L/R/T/B。

动态状态49→45维，编码器输入65→61，参数895123→894867（仅减少256个输入权重）。除了这个必要的输入尺寸变化，S0前端、12层协调主干、pair bias、额外关系value psi、共享布局头、原其他状态反馈均保持E142。无新空白反馈、形状试移、布局头改造或psi剔除。

从头scratch seed0，不加载旧权重；通过保留E142随机初始化流和删除相应输入列，使其余初始参数与E142逐张量相同。原36k/train500/dev500与共享缓存、batch64/末批32、FP32、增强、AdamW/LR/早停、数值门禁/模块更新投影、4h/200epoch上限不变。全部模块预测，原面积/无MIB，不读test。

监督/Loss完整沿用E139/E142：S2/S6去全部直接overlap，S8/S10/S12完整，权重.05/.05/.05/.05/.8。无阻隔Loss，无G/B×4，不增加直接监督层。新几何摘要仍no_grad，编码器参数正常学习，旧pair/state_embed路径保持。

启动入口：`python -u -B -m experiments.E143_directional_travel.run_queue --start`。本轮已授权后台执行：train→冻结唯一own-Loss-best→原固定dev500/E98.depth_budget.v1后处理→原规格独立硬检查→[Q]日志/与E142对比。Q不选权重，不追加seed或预算，错误/STOP阻断队列、不自动重试。

状态：queue_state.json、runs/e143_directional_travel_seed0/heartbeat.json；日志background/queue.stdout.log及launch_logs；Q摘要background/quality.log，原结果在该run/evaluation_no_mib_dev500。安全暂停后才显式--resume，保存全部恢复状态和累计预算；历史STOP、源码、权重、结果均保留。未来展示包含真实S0。

单seed、自适应停止与略微变化的编码器输入规模仍限制因果结论。详细数学定义和验证范围见DESIGN.md。
