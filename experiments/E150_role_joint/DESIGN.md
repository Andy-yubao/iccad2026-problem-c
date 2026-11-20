# E150 冻结定义

模型返回[13,B,N,3]，索引0为角色初稿，1–12为真实几何层。初始布局器复用E149结构；PublicProblemEncoder、FunctionalUnit和pair/形变反馈从E148独立复制，不跨实验导入私有生产算法。初稿隐藏不进入后段，只传同一公开面积归一后的center_x/center_y/log(w/h)布局。后段纯题目条件不使用初稿隐藏或reference。

整个681141参数模型联合scratch seed0。initial_generator包含角色/端口两块；problem_encoder保留原两块纯题目交互；units[0/1/2]各两个不同子层，每个单元调用两次，跨调用共享完整参数。无冻结模块或历史checkpoint迁移。动态49+16摘要仍detach，原布局/pair/state embedding形成的端到端梯度保留。

L=1/15*F_no_overlap(L0)+1/15*F_no_overlap(L4)+1/15*F_full(L8)+.8*F_full(L12)。F沿用原基础项系数；no_overlap重建保留项之和，不先加后减大数；full保留全部碰撞组。训练/监测/best/LR/早停均按此联合目标，不按某层单项另选权重。

features/batches/ports及role_features与E149逐字节相同，原共享CPU缓存key不变。role_cache.py只读登记的E149角色缓存，核对producer hash、成员及chunk sha；不构建或覆盖。轴交换同时变换端口坐标，纯图结构角色和hop保持不变。

本轮只记录Loss并冻结own-joint-Loss-best。所有模块/F/P全通道预测、不mask不覆盖、原面积/无MIB、test封存、Adam前门禁及累计预算恢复规则保持。
