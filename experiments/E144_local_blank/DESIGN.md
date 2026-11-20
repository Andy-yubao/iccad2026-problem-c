# E144：有上限的脱离距离与局部空白反馈

用户授权准备通过后后台从头训练并自动验证原固定500例。本组以E142为独立对照，不叠加其他两组修改。

45 dynamic features: capped axis escape / corresponding side, newly entered .25-side strip blank fraction / predicted area; remove old probe025/probe100; remaining E142 fields unchanged

Original pair18 attention bias retained. Original receiver-aware relation values retained.

Original E142 hidden and layout propagation retained.

其余完整沿用E142：原36k/cache、seed0、batch64/末批32、FP32 deterministic、原优化器/数值门禁/早停、每组4h/200epoch上限。E139监督：S2/S6不含overlap，S8/S10完整，各.05；S12完整.8。原reference及公开硬约束Loss不变，不读test。网络预测全部模块，由原后处理满足硬规格。

训练、监测、best、LR、早停使用同一Loss，Q不参与选模。完成后冻结own-Loss-best，自动执行原固定dev500及原后处理预算、独立硬检查，立即在评价日志、队列日志和background/quality.log输出[Q]摘要。STOP/错误阻断后续，不自动重试或恢复。

实时记录：preflight.json、queue_state.json、runs/e144_local_blank_seed0/heartbeat.json、launch_logs/。本轮整体入口为E144_local_blank.background.run_suite，依次E144→E145→E146。

所有动态摘要只取当前预测及公开输入，detach；布局的原state_embed/pair梯度保持。原两层初始化保留；E144/E145继续传递其隐藏输出，E146仅通过S0布局向后传递信息。

## 本轮几何定义

令当前预测矩形为R_i、宽高为w_i/h_i。方向L/R的对应边长为w_i，T/B为h_i。

轴向脱离距离只对当前重叠的有效j≠i取最大值，固定形状/其他矩形，分别为x_i+w_i-x_j、x_j+w_j-x_i、y_j+h_j-y_i、y_i+h_i-y_j。非重叠对不参与。输入min(D,对应边长)/对应边长∈[0,1]，不再施加穿透阈值或到外框距离上限。没有当前重叠时为0；它不代表移动路径安全。

空白输入：向该方向移动0.25倍当前对应边长后，新进入条带在起点当前预测外框内、未被其他有效矩形覆盖的面积，除以当前预测w_i*h_i，范围[0,.25]。不统计平移前后重合的75%自身范围，不用public目标面积作分母；外框外不算空白。其他矩形先求覆盖并集，重叠不重复扣除。自身不与新条带内部相交，可共同参加并集构建。

用所有当前矩形的x/y边坐标压缩网格，两个轴向覆盖指示矩阵相乘恢复覆盖并集，再用二维前缀，再用面积前缀的双线性查询积分条带；不是栅格近似。面积积分内部FP64以减小小条带的前缀相减误差，输出及网络训练保持FP32；不引入reference。排序/并集均在GPU，O(BN²)存储。覆盖计数为0/1乘加，N≤120时FP32精确，保持deterministic/TF32关闭，避免Windows确定性scatter的图捕获同步；计数矩阵乘法O(BN³)计算，准备阶段测量实际耗时。

原自身8维、group18维、boundary11维保持；方向脱离4维、空白4维，共45动态+16静态。旧0.25s和1.0s重叠变化响应删除，以新空白4维替换短响应的位置。参数894867，原E142对应权重保留同seed scratch初始化。
