# 延长预算后的候选7：C5完整求解监督与精确WH

用户将整个窗口截止延至北京时间17:45，原05:04:02UTC起点/8候选上限不变。C6以600.19秒预算结束并保留epoch0，整个模型严格等C5，复用已有C5评价。新增时间用于不同监督来源，不改旧候选、教师16500或活动run身份。

新公共教师从C5 unique own-Loss best +原完整E85 CPU1生成，单GPU/单CPU数学线程、单worker串行，离线最多1200秒，全部计入总墙钟。新train4000为原16k内部固定子集：强制纳入原train_monitor500和原五train实施ID，其余default_rng(730)无放回补齐再排序；原split和冻结索引不改。源C5已暴露原16k/继承更大来源，不能称新独立4000泛化。

开发500直接复用C5保存的full最终坐标和硬检查，不新跑dev solver/scorer；只重建公开NN base/source状态以对齐实际部署输入。保存每例精确float64最终位置、归一化实际WH、center/logaspect、实际部署L/F状态、member/global-index/来源哈希/CPU求解费用；近邻和教师轴选择使用实际WH及保存中心，避免将教师容差面积错误重构为nominalWH。网络仍是公开规格+允许草稿，exact WH只Loss。网络输出名义面积与teacher容差面积仍不完全可达，aspect>5和面积差明确统计保留。

F-only从C5严格暖启动，241547原F读出全部保留、E74冻结；student.DeploymentModel不变，无新增推理/CPU步骤。固定新Loss承接C5公开N权重/同组自由轴/公开boundary+MST权重3.687645，仅新teacher几何用于近邻/教师轴判定；同一总函数反传/train500/dev500/best/LR/早停。新4000子集、教师来源、WH数值路径及LR是登记迁移，不单因子归因。fresh AdamW seed0/batch32 FP32/noAMP/TF32，LR .0001/.00005/.00002/.00001，最多600秒/100轮和09:45UTC截止；完整best/latest Adam/RNG/order/cursor/seen/调度/预算保留。

teacher完成且来源/成员/分块hash核验后，原五train进行真实Dropout/Adam普通vs图更新、完整恢复下一步、非有限target及WH在Adam前拒绝、L冻结/NN输入隔离/语义门禁，成功后才计正式第7槽。唯一自身Loss best一次原full CPU1 dev500、新raw2000、同100图；若整个epoch0等C5则复用完整C5终末/raw/图。流水线只等待teacher，无并行训练或正式计时；不自动重试错误/缺失标签，不按Q挑checkpoint。测试封存。
