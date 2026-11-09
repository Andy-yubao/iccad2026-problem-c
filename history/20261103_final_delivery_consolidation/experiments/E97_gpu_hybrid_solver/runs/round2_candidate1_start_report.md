# 第二窗口首个候选：C3公开边界/分组任务

用户恢复4小时长任务；本窗口固定UTC2026-11-02T05:04:02至09:04:02（北京时间13:04:02至17:04:02），全部工程/训练/评价/绘图/等待计入。此前study.json保持关闭，新账本round2_study.json，最多8个正式候选，单个计算任务/CPU数学线程1、单GPU串行，自动唤醒每30分钟已恢复，测试封存。

保存结果只读分解：C3/C6完整Q超额中的77.30%/75.61%来自软约束，主要是公开边界和分组；线长/外框较次要。gap诊断显示大部分是实际空间间距，不能靠微小数值接触修补解释。首次gap诊断仅在JSON序列化numpy int32失败，失败源码保留；v2只修正标量序列化，未重新推理、求解或评分。

本窗口单线程原五train分阶段检查：当前旧C3公开输入到draft均时10.03ms/P95 11.67ms；把相同decode/scaling捕获到GPU图为9.35ms/P95 9.80ms，物理输出逐例逐次完全相同。该工程收益约.68ms，不能把此前80–101ms到当前10ms的环境变化归功于它，更不是完整服务达标证据。

为保持本窗口条件一致，只登记一次C3完整单线程基线（新捕获路径）：完整Q1.013366283，500合法、0行回退、均时.210807秒/P95 .526127秒，135例超过.25秒；500例输入神经草稿逐例完全相同于旧C3记录，raw2000复用旧C3，同100最终图已完成。原CPU后处理预算与强度未改变。

本候选严格暖启动C3 unique own-Loss best SHA d784e96f46decf7a487c25bde2ab16db8cf6ebae1bc0632434cc5e2f78173fce，241547个F参数全部读出保留，E74冻结，部署网络不增层。fixed Loss=原C3 center1+shape.25+near-relative.25+physical-clearance-tail1，再加3.264434296319982×public_soft。public_soft含每块最坏所需边界侧的SmoothL1，以及teacher-native Kruskal组内接触树的法向贴边/正切向覆盖损失，按公开N_soft归一化。teacher只用已有公共求解器标签，不用自由reference或评分分母，不进NN。树标签是软监督，不保证全局可行；nominalWH/教师面积容差、9个train aspect>5局限保留。

系数按预登记train_monitor500一次局部状态梯度标定：500活跃成员，base中位梯度.841728956，task .128924169，预定份额.5，系数3.264434296；0优化器、0solver、0dev或Q调用。Loss保持逐例无N加权算术均值，同一函数用于反传、固定train/dev500、best、LR与早停。系数文件SHA 876a4e1314b602810594cbd2e2f7209f6d265bc931519ab6e0c030496d57eb4c。

原五train预检35.48秒通过：真实Dropout/Adam eager/图参数与动量差0，图捕获无参数/Adam更新，重建恢复下一步model/Adam/RNG逐位一致；非有限teacher在Adam前拒绝，L逐位冻结，初始模型完全等于C3，teacher接触标签从NN输入中隔离，接触/分离与边界语义反例通过。fresh AdamW seed0，batch32 FP32/noAMP/TF32，LR .00025/.0001/.00005/.00002，最多1小时/100轮及原本窗口截止，完整best/latest恢复状态保留，初始模型由受保护的C3源权重保留。

后台pipeline已启动，见round2_candidate1_launch.json/stdout/stderr。训练结束仅自身Loss唯一best一次原完整E85单线程dev500、新几何raw2000和同100图。若best为epoch0且整个模型严格等于来源，复用本窗口已完成C3完整基线、其同100图和原C3 raw2000，不再相同求解。无Q反选checkpoint、无dev系数扫描、无额外CPU增强。

活动训练器/模型/Loss/计划/评价依赖已绑定冻结；普通进程查询看不到后台不代表退出，依据日志、锁、权重和状态判断。任何数值失败先保存诊断，不无限重试。到17:04:02北京时间或候选上限即安全收尾关闭heartbeat，目标未达不得称成功。
