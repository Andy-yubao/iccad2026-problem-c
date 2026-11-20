# E147：只在监督层之后清除隐藏状态传递

2026-11-17 用户确认定义并授权准备完成后后台启动。E142为独立对照；从头训练，不加载已训练权重。

只在S2→S3、S6→S7、S8→S9、S10→S11清除上下文隐藏状态。下一层由公开静态属性编码＋上一层布局的state_embed＋当前需求/pair几何重建输入；不使用S0隐藏状态替代。其他层间照常传布局及隐藏状态，S0→S1也保留。S12无下一层。

每段内部照常积累隐藏状态：S0–S2；S3–S6；S7–S8；S9–S10；S11–S12。布局始终传递，不detach，保留通过布局、pair和state_embed的梯度；它不是截断反传实验。动态几何摘要保持E142原49维及原detach策略。

原两初始化块、12层width64、模块需求编码、pair18 bias、ψ关系value、局部消息、残差、FFN和共享读出头全部保留；895123参数，逐值对应E142同seed scratch初值。

配置复用E142：冻结36k及现有CPU缓存、seed0、batch64/尾批32、确定性FP32/TF32关闭、原增强/AdamW/更新门禁/LR/早停、4h/200epoch上限。E139监督不变：S2/S6去直接overlap；S8/S10完整，前四项各.05；S12完整.8。其他层不加直接监督；不读test。

训练、监测、best、LR、早停使用同一Loss。训练正常结束后冻结唯一own-Loss-best，自动原固定dev500、原后处理预算及独立硬检查；立即在评价日志、后台日志和background/quality.log打印[Q]，不按Q选权重。

入口：python -u -B -m experiments.E147_supervision_hidden_reset.run_queue --start。实时状态在queue_state.json和runs/e147_supervision_hidden_reset_seed0/heartbeat.json；后台日志background/queue.stdout.log及queue.stderr.log。STOP/QUEUE_STOP/异常阻断后续，不自动重试或追加实验。

## 实现

static=input(public_features)。初始化h经原两图块得到S0。循环生成S_depth时，仅若depth-1属于监督层集合，选择context=static，否则context=h；再调用原layer(context+state_embed(previous_layout), current_demand, pair)。监督集合含12但没有S13，故实际只有四处reset。不增加随机参数，不复制/缓存S0上下文，不改变原布局读出与几何计算。

## 准备检查

固定各读出的真实布局，对S0及S1–S11隐藏输出逐个施加扰动，检查下一层输入：只有监督层后的四处完全不受影响，其余八处必须响应。四处reset后下游输出与未扰动运行逐位一致。核对S1/S2与同参数E142逐位一致；验证最终目标仍可通过布局通路向四个reset层读出反传。短检使用原登记五训练例，丢弃全部短检更新，正式重新scratch；不提前读取dev/test载荷或做后处理。
