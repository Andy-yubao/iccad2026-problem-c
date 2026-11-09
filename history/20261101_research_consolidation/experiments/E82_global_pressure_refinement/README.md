# E82：冻结修正模型，补充全局公共通道信息

第三阶段第2候选，沿用用户对模块化长任务的自主训练授权，中央账本仍为[E81/study.json](../E81_multisource_refinement/study.json)。前一组E81已完成27轮/best26，部署171.3万参数、dev交面积2.491%、Q1.016242、最大1.825773；[复核](../E81_multisource_refinement/runs/completion/inspection.md)保留，不重训或重评。

失败分析表明平均交面积和A下降后，164仍29/29主修复失败，329新出现27/27失败和救援，全500救援4→6，固定形状证书432例失败。这不等于实际SOCP全部无解，也不能推出容量不足。新假设是：局部几何消息和标量Loss没有显式提供全局锚点路径区间；在节点输入中补充公共全局信息，可能帮助既有F作更合适的局部修正。

完全冻结E81两个布局器及整个F的497个张量，只训练三个`4→192`零初始化输入嵌入，共2304权重。在各原节点Linear之后、SiLU/LayerNorm之前相加，初始输出保持E81。旧F的Dropout始终eval；新嵌入Dropout.05训练时启用。没有新增随机修正阶段，也不继续无依据加宽。F总243851参数、部署1715390；训练模型含辅助P74为7545329，P74不进入部署计算。

四个新特征只从冻结布局器公开草稿、公开预放置和自身预测框计算：采用源mode0轴/先后次序，16次正向max下界与反向min上界传播，取`(required_lower-xy, xy-required_upper)/sqrt(public area)`，截断至[-8,8]，padding为0。正值表示通道压力，负值表示剩余间隙。上下界在每来源初始草稿上只算一次，三个F阶段共享；不读取reference/标签，不调用求解器，context无梯度、不对离散次序做STE。原草稿形状尚未MIB协调，因此特征可能不准；截断传播并非合法性证明。额外计算可能增加耗时。

Loss与E81完全相同：每来源`G+4I+5E+.25A`，每例`mean_source+.25*(max_source-mean_source)`；训练、train/dev500、完整dev和选模同一定义，从epoch0无渐入。旧G八项保留，A仅为辅助形状压力。Fresh AdamW只训练新2304权重、clip1/decay.0003，原四档LR/早停、FP32/seed0/batch32不变。每个冻结32k成员同时有E42/E74两种公开草稿；原dev2000/train-dev500不变，train500仍旧16k来源，测试封存。

预检用同5个train成员，含2新增半数，检查初始E81函数/上下界符号/公开输入/冻结全部497张量/规格面积框/固定Loss/CUDA/非有限值/真实新路径Dropout Adam恢复及流水线入口。单组最多2小时200轮；本阶段最多3候选6小时，全局18小时与此前费用保持。`runs/STOP`安全停止，恢复不得清零预算或状态。

通过预检后独立环境入口：`python -u -B -m experiments.E82_global_pressure_refinement.pipeline`，恢复加`--resume`。原pipeline自动唯一own-Loss best完整dev2000、一次P42部署flash500、同100例reference/current/E42图和报告，导出部署与独立F。已有E81/E80/E42结果只读复用，不额外终末评分P74，不按Q换checkpoint；E42/E50继续冻结。输出：[runs/20261029_e81_pressure_seed0](runs/20261029_e81_pressure_seed0)。源码首次训练前冻结，诊断放runs；不得运行准备脚本覆盖已运行实验。

已完成28轮/best12并深度复核：[完成审计](runs/completion/inspection.md)。同Loss下降、交面积2.445%略改善，但加权Q1.018035和最大2.056040回退；本配置不升级，保留E81作为主要紧凑候选。当前无活跃训练，本组不恢复不重评；下一阶段先分析目标与修复错位。
