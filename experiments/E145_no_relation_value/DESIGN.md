# E145：移除关系 value 消息

用户授权准备通过后后台从头训练并自动验证原固定500例。本组以E142为独立对照，不叠加其他两组修改。

Current decoded layout/public specs only; detached geometric summaries, shared demand encoder, per-layer fusion; see DESIGN.md

Original pair18 attention bias retained. Ordinary attention values only; remove psi network and its parameters.

Original E142 hidden and layout propagation retained.

其余完整沿用E142：原36k/cache、seed0、batch64/末批32、FP32 deterministic、原优化器/数值门禁/早停、每组4h/200epoch上限。E139监督：S2/S6不含overlap，S8/S10完整，各.05；S12完整.8。原reference及公开硬约束Loss不变，不读test。网络预测全部模块，由原后处理满足硬规格。

训练、监测、best、LR、早停使用同一Loss，Q不参与选模。完成后冻结own-Loss-best，自动执行原固定dev500及原后处理预算、独立硬检查，立即在评价日志、队列日志和background/quality.log输出[Q]摘要。STOP/错误阻断后续，不自动重试或恢复。

实时记录：preflight.json、queue_state.json、runs/e145_no_relation_value_seed0/heartbeat.json、launch_logs/。本轮整体入口为E144_local_blank.background.run_suite，依次E144→E145→E146。

所有动态摘要只取当前预测及公开输入，detach；布局的原state_embed/pair梯度保持。原两层初始化保留；E144/E145继续传递其隐藏输出，E146仅通过S0布局向后传递信息。

普通attention为m_i=Σ_j α_ij V_j；仅移除ψ消息的edge/receiver/sender/relation_value网络与其参数。原pair18 attention bias、局部wire/group消息、需求融合、QKV、残差、FFN和跨层隐藏/布局传递不变。参数869779（比E142少25344）。先按E142顺序初始化全部参数再删除ψ，不改变保留参数的随机初值。
