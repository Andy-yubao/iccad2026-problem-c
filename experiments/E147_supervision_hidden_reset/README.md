# E147：只在监督层之后清除隐藏状态传递

2026-11-17 用户确认定义并授权准备完成后后台启动。E142为独立对照；从头训练，不加载已训练权重。

只在S2→S3、S6→S7、S8→S9、S10→S11清除上下文隐藏状态。下一层由公开静态属性编码＋上一层布局的state_embed＋当前需求/pair几何重建输入；不使用S0隐藏状态替代。其他层间照常传布局及隐藏状态，S0→S1也保留。S12无下一层。

每段内部照常积累隐藏状态：S0–S2；S3–S6；S7–S8；S9–S10；S11–S12。布局始终传递，不detach，保留通过布局、pair和state_embed的梯度；它不是截断反传实验。动态几何摘要保持E142原49维及原detach策略。

原两初始化块、12层width64、模块需求编码、pair18 bias、ψ关系value、局部消息、残差、FFN和共享读出头全部保留；895123参数，逐值对应E142同seed scratch初值。

配置复用E142：冻结36k及现有CPU缓存、seed0、batch64/尾批32、确定性FP32/TF32关闭、原增强/AdamW/更新门禁/LR/早停、4h/200epoch上限。E139监督不变：S2/S6去直接overlap；S8/S10完整，前四项各.05；S12完整.8。其他层不加直接监督；不读test。

训练、监测、best、LR、早停使用同一Loss。训练正常结束后冻结唯一own-Loss-best，自动原固定dev500、原后处理预算及独立硬检查；立即在评价日志、后台日志和background/quality.log打印[Q]，不按Q选权重。

入口：python -u -B -m experiments.E147_supervision_hidden_reset.run_queue --start。实时状态在queue_state.json和runs/e147_supervision_hidden_reset_seed0/heartbeat.json；后台日志background/queue.stdout.log及queue.stderr.log。STOP/QUEUE_STOP/异常阻断后续，不自动重试或追加实验。
