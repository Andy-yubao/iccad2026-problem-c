# 给接手模型的导读

2026-11-08：E114三组与E115已完成，用户选择4+12为后续主要基线（[v2](../protocol/research_baseline_v2.json)）。加权Q/尾部坏例、四模型训练用时及单例入口干预见[最新诊断](../output/supervision_q_analysis_20261108/report.md)；[4+12固定5例＋最差5例三相图](../visualization/E114_supervision/20261108_staged_three_phase10/index.html)。全500结果与旧实验证据保持，不重训、不读test。

状态截点：2026-11-05。目标是帮助你基于已有证据形成判断，不要求维护旧解释，也不授权新实验。先读项目README、protocol和workspace_structure，再读本页；需要细节时进入[研究历史](research_history.md)。

## 先建立任务模型

**后续新实验规则已于2026-11-06更新：** 原始面积/规格保持，MIB不修正、不作为学习/修复目标、不罚分；新默认独立随机36k/2k/2k集合，完整分片重新划分，不沿用旧筛选。下文MIB平均面积及旧split说明仅解释历史。[现行数据与评分边界](protocol.md#2-数据与信息边界) · [E108准备状态](../experiments/E108_random_reference40k/README.md)。本次仅授权后台准备数据及图片，未授权训练或测试评价。

输入是数量可变的模块、面积、线网/端口、边界、分组/MIB、固定尺寸与预放置规格。网络输出通常是中心cx/cy与log宽高比，面积由解码保持；固定规格另处理。主线不是直接输出图像，而是公开问题 → 布局器L → 可选修正器F/内部几何 → 后处理H → 独立硬检查。E102才是独立图像路线。

训练可用reference监督，推理不得读自由reference或评分分母。主线采用公开MIB平均面积派生任务；原reference未必与变换后的规格完全相容。网络的端口输入现行完全压缩为逐模块总权重/加权中心；精确目标仍用完整连接。不要把旧完整端口实验重新提成从未尝试的方向。

评价中的本地Q越低越好，以官方质量/软违规计算但将时间因子置中性；它不是正式比赛分数。加权汇总偏重大N；平均Q与加权Q可能反向，近期研究另固定混合Q为两者均值。完整逐例延迟另报，不能以GPU草稿时间或批量吞吐替代。

## 五分钟必须带走的认识

1. **完整后处理仍承担硬合法与大部分耗时。** E97曾把三步弱修复当共同前提，后来相同草稿接回完整后处理，Q和候选排名大幅改变。这是已纠正的重要研究判断，不要再次缺失质量基线。[原补测](../history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/fullrepair_singlethread_findings.md)。
2. **学习修正器有贡献，但更少重叠不是最终Q的充分条件。** E83排除了“全靠固定投影”的解释；E84/E82等又提供几何与Q背离的对照。[归因](../history/20261101_research_consolidation/experiments/E83_strategy_attribution/runs/inspection.md) · [上游/下游核查](../history/20261103_final_delivery_consolidation/experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit/report.md)。
3. **历史C5、重训C5不是同一权重。** 原C5 dev500加权Q1.012685893；首次64k暖启动1.031581961；历史链路重跑1.018307314。另一个三阶段均64k链路在63561步停止，不能写成完成。[E101身份表](../experiments/E101_final_c5_solver/README.md)。
4. **训练代理→可用方向→真实模型更新→后处理Q是四道门。** E103排序略升、全局相关性略升，续训混合Q只小幅变化，尚无配套原Loss续训控制；E106数值预测略有用，但方向检查47.06%，接近随机。[E103](../experiments/E103_offline_loss_alignment/README.md) · [E106](../experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/result_review/report.md)。
5. **扩容与结构并非没有收益，但现有证据不支持单因素归因。** 固定深度后处理下E98增强→E104修复→E105混合Q为1.047137→1.037447→1.033255。E104中途改forward，E105是结构组合、单seed；三者草稿仍0/500硬合法。[E104](../experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0/evaluation_depth_dev500/report.md) · [E105](../experiments/E105_gated_prenorm_placer/runs/20261105_gated128x16_seed0/evaluation_depth_dev500/report.md)。
6. **有限数值不等于训练健康。** E54出现输出截断死区；E95/E104有未归一化局部消息放大；E107是预设相对单例门禁触发，未证实NaN/Inf或持续几何崩坏。不要一律称“梯度爆炸”，也不要仅降LR就宣布根因解决。[机制历史](research_history.md)。
7. **无reference尚未被证明不可学。** 已失败的具体优化设置、截断平坦区和不可靠代理不构成不可能性证明。E71已试过密度项；仅再次添加密度不是新路线。最新公开能量/展开优化是未实施提案；用户否定了固定父稿2%总位移限制。[最新提案](../output/reference_free_optimization_proposal_20261105.md)。
8. **保持曝光和授权边界。** dev长期反复使用；E103/E106内部留出并非布局模型未见数据。E101旧test2000多次曝光不等于其他实验可以读测试。所有完成/停止任务不自动恢复；本次核查未发现Python进程。[状态](current_status.md) · [协议](protocol.md)。

## 二十分钟的最短证据链

依次读以下原件，不必遍历全部实验：

1. E97 [完整后处理纠正](../history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/fullrepair_singlethread_findings.md) 与 [第二轮收尾](../history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/runs/round2_final_report.md)：交付方法为何留下、目标为何仍未达。
2. E103 [全局Loss–Q](../experiments/E103_offline_loss_alignment/runs/20261104_global_independent2000_seed0/report.md) 与 [真实续训评价](../experiments/E103_offline_loss_alignment/runs/20261104_learned_v1_lr1e3_seed0/evaluation_depth_dev500/report.md)：统计可预测性和训练有效性的区别。
3. E105 [固定dev500](../experiments/E105_gated_prenorm_placer/runs/20261105_gated128x16_seed0/evaluation_depth_dev500/report.md)：最新完成布局器的身份与能力。
4. E106 [结果检查](../experiments/E106_postprocessor_aware_loss/runs/20261105_bank25000_seed0/result_review/report.md)：为何暂不能把代理接回训练。
5. E107 [失败诊断](../experiments/E107_geometry_posttrain/runs/20261105_geometry_ref10_seed0/diagnosis_and_revision_proposal.md)，再读[后续提案](../output/reference_free_optimization_proposal_20261105.md)：前一份末尾固定小范围建议已被否定，不能当现行决定。

若需要理解GNN/attention/旧模型细节，再读[2026-10-30全貌报告](../history/20261103_final_delivery_consolidation/output/2026-10-30_项目研究全貌报告/项目研究全貌报告.md)。它在E86时截稿，状态已过期；其概念解释和E42–E85归因仍有用。实际实验位置用[目录索引](experiment_catalog.md)，不猜原路径。

## 对下一次判断的要求

先明确你想解决的是组织能力、草稿几何、可修复拓扑、软约束、质量尾部还是耗时；不同目标可能需要不同证据。给建议时列出最强支持、最强反例、尚缺控制和最小能区分解释的下一步；不要把“未证实有效”写成“整体无效”。

新实验如获授权，应先固定科学问题、来源/成员、目标/选模、后处理口径及停止规则。优先复用已存坐标、trace和标签诊断，不靠反复重跑dev或test寻找方向；也不为追求形式完整而默认补大量控制和预防测试。

本次没有预设接手模型的结论：E101是保留交付，E105是近期固定深度链路的有用参照，二者不是已在同协议下决出的全局优胜关系。未来判断应由实际可比证据支持。
