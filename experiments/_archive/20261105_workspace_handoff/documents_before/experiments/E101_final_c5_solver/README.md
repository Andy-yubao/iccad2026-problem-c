# 最终交付：C5＋原完整后处理

用户于2026-11-03结束方法探索，选定E97第二轮C5架构及原完整E85单线程后处理，授权64k后台重训与训练结束后唯一冻结模型的2000例最终测试。本目录是独立交付实现，旧实验仅作为归档证据，不再追加候选或自动唤醒。

当前另获授权的新运行已经启动：**E42随机16k → E74后训练32k → 原C5 F16k → 冻结模型 → dev500 → 已登记test2000重复评价**。三个训练阶段均正常早停，无时间或轮数上限；使用原E85教师目标、原C3 F暖启动及原完整后处理。新入口为[完整方案与操作](runs/20261103_replay/README.md)和[后台状态](runs/20261103_replay/state.json)。新权重与原交付分开保存，测试不参与训练或Q选权重。

新链路的训练已全部完成：E42早停54轮/best51、E74早停45轮/best43、F早停18轮/best6。2026-11-04修复dev500评分器缺失上游导入文件的问题，原冻结模型及原训练源码完全不变，直接恢复dev500与随后test2000；[修复证据](runs/20261103_replay/evaluation_dependency_fix_20261104/README.md)。

用户2026-11-04进一步授权[三阶段均64k运行](runs/20261104_chain64k_seed0/README.md)，准备后自行后台启动，串行等待上一评价完成。E42随机64k→E74 64k→C5 F64k→冻结→dev500→原2000例重复测试；原E85教师目标扩展48k，精确磁盘缓存每批传输，原结构/Loss/正常早停/完整后处理保留。该运行完全独立，状态见其[state.json](runs/20261104_chain64k_seed0/state.json)。

原64k运行已完成训练及测试：L在31轮正常早停、best31，F在21轮正常早停、best12；原冻结权重SHA256为`c1f0ce288bfe8fee350e0ee92b29f9523487eb61753b741d71c18fafe350f04e`。原运行证据及其评价清单检查迁移均保留。**下文为原64k运行设计和命令记录，新历史链路以其独立方案为准**；本次2000例因此属于重复评价。

## 固定设计与训练

公开输入经逐模块端口总权重／加权中心压缩，进入192宽、12层布局器L；三段神经修正及原C5内部几何算子产生F草稿；原完整CPU单线程后处理生成最终解，独立硬检查核验。没有加入E99吸引排斥、学习关系头或E50 pro搜索。模型、Loss和后处理源码独立复制，来源见[provenance.json](provenance.json)。数据、评分、展示复用共享模块。

本次是权重暖启动的新运行，不是从随机权重训练，也不是旧运行普通续训。原C5资产保护于`assets/c5_baseline.pt`，SHA256为`028ea3aa124c8cd9e7c3cc7bf8587c699ab9b4932f256899e3780d4d8c8816e0`。完整登记见[plan.json](plan.json)。

1. **L阶段**：取原C5的E74布局器权重，使用冻结v6前64k成员；固定原E74的`G+0.5碰撞门控方向Loss`，保留原图聚合Linear冻结及每模块Adam实际位移0.5%上限。新AdamW，四档LR为2.5e-4/1e-4/5e-5/2e-5，最多200轮／26小时，正常早停优先。原24小时上限由用户2026-11-03明确延长2小时；安全保存后仅迁移预算，保留优化器、RNG、成员游标、调度和累计耗时，原计划/源码/best/latest另存于该run的budget_extensions中。
2. **教师与新草稿**：原C5＋原完整后处理仅从公开输入生成train64k/dev_monitor500教师；不读取评分分母、不按Q选教师。新L的唯一Loss best重新生成F的输入草稿。教师坐标转为中心／log宽高比，保留原C5名义面积重建限制；另存实际XYWH供审计。扩充目标来源由旧E85迁移到冻结C5，明确登记为新阶段。
3. **F阶段**：从C5修正器权重暖启动，冻结新L；使用原C5公开N加权、组内轴自由及公开边界／动态MST固定Loss。原系数和归一化不扫描。新AdamW，四档LR同上，最多100轮／12小时。

两个阶段都保持seed0、batch32、FP32、无AMP/TF32；train/dev固定各500监测。反传、监测、best、LR及早停使用同一逐例总Loss；best按dev500算术平均选取，epoch0允许胜出。每三次无实质改善降LR，最低档至少完成两轮且满足停滞条件后早停。best/latest保留优化器、RNG、成员顺序／游标、seen、调度及累计预算。触及上限如实标记budget，不宣称收敛。

用户随后要求优化GPU速度。L在634步、cursor20288处安全保存，显式迁移到第二段CUDA图执行AdamW和原逐模块步长保护；更新前数值门禁保留，每个LR档单独捕获，避免图保留旧LR。batch、FP32、Loss、成员顺序、Adam动量／步数、RNG、调度及预算均保留；GPU偏置校正存在已登记浮点次序差异，不称逐位一致续训。五例短检更新均时205.27→101.57ms，最大参数／动量／Loss差7.11e-6／1.93e-5／1.70e-6；未锁频，只是工程短检，不当作整轮加速倍数。F未测得收益，保留旧更新路径。原源码与best/latest归档于本run的engineering/v1_preserved，迁移见engineering/migration_v2.json。

64k成员、顺序及原split_v1不变；已有E42的16k、E74的32k与E97的16k曝光历史保留。共享E58有效缓存逐张量转换为相同FP32磁盘映射，分批固定形状传输，避免将64k整库复制进16GB显存。自由reference仅用于L监督，教师只作F监督，不进入网络输入。

## 评价与交付

完成L、F两阶段后，组合唯一Loss best，写入`delivery/model.pt`并在读取测试载荷前冻结哈希。先完整dev500和同100例reference/L/F/最终图，再按原冻结local_test2000顺序做一次最终评价。测试不决定checkpoint，不触发重训、系数或后处理修改。所有成员、坏例和超时例完整保留。

逐例完整计时含公开规格复制／遮蔽、MIB公开平均面积变换、特征、传输、L/F及内部几何、全部原后处理、必要硬检查与同步；磁盘加载、冷启动、评分和绘图单列。报告均值/P95/最大值和超过250ms数量；没有另测批量吞吐，不以其替代逐例延迟。Q为中性时间加权本地质量，且任务采用公开MIB平均面积变换，不是正式比赛成绩。v5排除旧v2已暴露测试成员，历史曝光和开发自适应限制仍见协议。

原64k运行：[状态](runs/20261103_final64k_seed0/state.json) · [L进度](runs/20261103_final64k_seed0/L/progress.json) · [F进度](runs/20261103_final64k_seed0/F/progress.json) · [原测试报告](runs/20261103_final64k_seed0/evaluation/local_test/report.md)。新运行完成后另在`runs/20261103_replay/report.md`汇总。

## 操作

新后台入口为`runs/20261103_replay/run.py`，已经启动，不需要重复启动。新运行的停止/恢复说明见其README。原64k入口`python -u -B -m experiments.E101_final_c5_solver.pipeline`仅保留作历史记录。完整日志保存在各自运行目录；出错保存并退出，不自动循环重试。

推理接口：`from experiments.E101_final_c5_solver.solver import Service`；使用`Service(checkpoint, example_public_inputs)`初始化常驻GPU和原生后处理，再调用`service.solve(public_inputs)`。网络和求解API均不接收reference或评分分母。
