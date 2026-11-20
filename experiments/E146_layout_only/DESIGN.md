# E146：层间仅传布局，不传隐藏状态

用户授权准备通过后后台从头训练并自动验证原固定500例。本组以E142为独立对照，不叠加其他两组修改。

Current decoded layout/public specs only; detached geometric summaries, shared demand encoder, per-layer fusion; see DESIGN.md

Original pair18 attention bias retained. Original receiver-aware relation values retained.

Only prior layout crosses layers. Each layer starts with public static embedding + prior-layout state embedding, current demand and pair geometry. No previous contextual hidden or S0 hidden bypass; no detach.

其余完整沿用E142：原36k/cache、seed0、batch64/末批32、FP32 deterministic、原优化器/数值门禁/早停、每组4h/200epoch上限。E139监督：S2/S6不含overlap，S8/S10完整，各.05；S12完整.8。原reference及公开硬约束Loss不变，不读test。网络预测全部模块，由原后处理满足硬规格。

训练、监测、best、LR、早停使用同一Loss，Q不参与选模。完成后冻结own-Loss-best，自动执行原固定dev500及原后处理预算、独立硬检查，立即在评价日志、队列日志和background/quality.log输出[Q]摘要。STOP/错误阻断后续，不自动重试或恢复。

实时记录：preflight.json、queue_state.json、runs/e146_layout_only_seed0/heartbeat.json、launch_logs/。本轮整体入口为E144_local_blank.background.run_suite，依次E144→E145→E146。

所有动态摘要只取当前预测及公开输入，detach；布局的原state_embed/pair梯度保持。原两层初始化保留；E144/E145继续传递其隐藏输出，E146仅通过S0布局向后传递信息。

用户最新更正优先：不能把S0隐藏状态作为各层共同底座。令s_i=input(public_features_i)为无上下文的公开静态编码；原initial_blocks(s)仅生成S0布局。每个精修层独立接收s+state_embed(上一层布局)，以及由上一层布局产生的demand/pair；不接收前一精修层输出h，也不接收initial_blocks产生的h0。s可共享计算，但没有经过任何初始化attention块，既不是S0隐藏状态，也不是前层隐藏状态。

每层内部attention、ψ、残差、FFN及原全部49维反馈不变。上一层布局不断传递且保留原梯度通路，不做detach；本组不是梯度截断实验。层内模块隐藏向量仍正常通信，只有跨层的隐藏旁路移除。参数895123，与E142逐值对应同seed scratch初值。
