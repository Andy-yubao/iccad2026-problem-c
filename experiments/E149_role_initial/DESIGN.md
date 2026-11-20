# E149 注册定义

## 初稿唯一输出

初稿网络输入模块公开16维特征、结构角色16维、引用端口位置/总权重/连接数量4维及静态关系；两个attention块之后只读出模块center_x/center_y/log(w/h)，返回[1,B,N,3]。不实例化任何几何更新网络。端口是固定公开锚点，只在key/value及局部关系聚合中提供信息。

角色字段顺序以role_features.NAMES为准。无权度排除self、零权和padding；B2B正权图求分量。最短路在B2B及P2B组成的图上为模块source求精确无权最短hop；group不作电气边。到P的距离只看constraints[:,1]，不把只有固定形状的F当位置锚点。无可达锚点时距离值0、有效标记0；P自身距离0、标记1。有限hop经过asinh压缩但不封顶。

端口使用原CPU缓存中的完整public pin_coordinates/pin_weights/pin_valid恢复。所有被引用坐标保留；相同位置合并，权重累加，线长语义不变。原缓存不保存未被任何模块引用的端口，本实验也不凭空引入它们。端口排序不作身份编码，模块/端口排列同时作用于关系矩阵后，模型等变。

关系输入6维：log1p正连接权、正边标记、同组标记、asinh(hop)、可达标记、self标记。每块用6→24→4的bias MLP，叠加内容attention。module query读取module/port key/value；局部聚合采用B2B/P2B行归一连接及单独G行归一关系。端口token不跨层更新，不构造port-query attention。

## 静态缓存与增强

基础features.py/batches.py/ports.py保留与E148完全相同的字节，兼容原共享缓存身份。新的输入白名单由public.py提供，不改旧基础缓存生产者。role_cache_v1只存public role_features、port_features、port_edges、int16 hop矩阵和成员，不含target/reference。identity绑定原缓存key与role_features.py源码hash；只构建缺失字段，已有身份不符报错。模型、optimizer和增强结果不进入静态缓存。

CPU组批把模块放在前N个槽位、端口放在后M个槽位，精确保存所有引用端口；padding和不可达明确标记。只搬运module-query×(module+port)的紧凑关系矩阵。训练x/y交换同时变换原公开特征、规格、标签、pin坐标和port token坐标；纯图结构角色/最短路不随轴交换改变。

## 目标与训练

初稿唯一Loss=原基础weighted分项之和，排除overlap_weighted；不通过先加后减大数实现。碰撞的三个直接分量均不反传；碰撞和full_total数值保留只作诊断，best/LR/早停均按initial no-overlap total。公开规格及reference监督按原定义，F/P全部预测。原模块更新保护、CUDA64和eager尾批32、AdamW、随机排列、增强和累计预算恢复机制继承。正式seed0重新初始化，不复用短检更新。

整个实验只训练一个候选、上限4h/200epoch；后续只有冻结与Loss记录，不调用任何后处理、Q或test读取。
