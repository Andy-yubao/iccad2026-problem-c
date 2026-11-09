# 全部实验实际目录索引

生成：2026-11-05。只列实际实验根，不展开run隔离副本、资产或数据。目录存在不等于完成训练；状态与结论见[当前状态](current_status.md)、[研究综合](research_history.md)。同一编号可在多次归档中有不同阶段副本，不静默选择“最新版”。

历史README保留原相对路径和命令，可能需要对应manifest解释；不直接执行旧启动入口。

## 当前保留路径

| 实验 | 原标题 |
|---|---|
| [E116_first_draft_comparison](../experiments/E116_first_draft_comparison/RUNNING.md) | 原结构前4层 vs 四个隐藏Attention＋门控FFN、唯一全局条件读出；16k/最终Loss，获授权后台串行训练 |
| [E115_first_head_supervision](../experiments/E115_first_head_supervision/README.md) | 36k九图块、大首头1+9监督；队列先恢复E114 4+12 |
| [E114_supervision_placement](../experiments/E114_supervision_placement/README.md) | 64维12图块小模型：末层/密集/4与12层监督，仅准备未训练 |
| [E112_deep_geometry_budget](../experiments/E112_deep_geometry_budget/README.md) | 8图块＋更深几何关系MLP，近A参数预算 |
| [E113_deep_head_budget](../experiments/E113_deep_head_budget/README.md) | 8图块＋更深共享预测头，近A参数预算 |
| [E109_deep_supervision_shared](../experiments/E109_deep_supervision_shared/README.md) | A：共享预测头逐层监督，原E98条件 |
| [E110_deep_supervision_heads](../experiments/E110_deep_supervision_heads/README.md) | B：独立预测头逐层监督，原E98条件 |
| [E111_progressive_depth](../experiments/E111_progressive_depth/README.md) | 4→8→12层，只训练新四层与共享头，末层Loss |
| [E108_random_reference40k](../experiments/E108_random_reference40k/README.md) | E108：独立随机40k原始reference数据及图片准备，无训练 |
| [E98_loss_weight_learning](../experiments/E98_loss_weight_learning/README.md) | E98：已停止的调权方向与受保护基线 |
| [E101_final_c5_solver](../experiments/E101_final_c5_solver/README.md) | E101：原C5交付与重训证据 |
| [E102_reference_image_generation](../experiments/E102_reference_image_generation/README.md) | E102：题目输入到 reference 风格图片 |
| [E103_offline_loss_alignment](../experiments/E103_offline_loss_alignment/README.md) | E103：Loss对齐的离线与真实训练研究 |
| [E104_scaling32k](../experiments/E104_scaling32k/README.md) | E104：32k、128×16 泛化增强版扩容实验 |
| [E105_gated_prenorm_placer](../experiments/E105_gated_prenorm_placer/README.md) | E105：方案一，完整前置归一化＋残差门控＋SwiGLU |
| [E106_postprocessor_aware_loss](../experiments/E106_postprocessor_aware_loss/README.md) | E106：后处理感知的可微Loss |
| [E107_geometry_posttrain](../experiments/E107_geometry_posttrain/README.md) | E107：E105局部几何后训练，reference10% |

## 20261018_E10_consolidation

| 实验 | 原标题 |
|---|---|
| [conflict_local_repair](../history/20261018_E10_consolidation/experiments/conflict_local_repair/README.md) | E5：冲突驱动局部重构探针 |
| [graph_proposal](../history/20261018_E10_consolidation/experiments/graph_proposal/README.md) | E1：图条件布局提议 |
| [intent_aware_repair](../history/20261018_E10_consolidation/experiments/intent_aware_repair/README.md) | E6：面向可修复布局的生成与意图感知修复 |
| [masked_construction](../history/20261018_E10_consolidation/experiments/masked_construction/README.md) | E3：带规则候选的逐步构造 |
| [repair_friendly_draft](../history/20261018_E10_consolidation/experiments/repair_friendly_draft/README.md) | E7：降低修复难度的布局草稿（已终止） |
| [topology_geometry](../history/20261018_E10_consolidation/experiments/topology_geometry/README.md) | E2：分离关系 → 固定形状几何 |
| [E8_fast_robust_postprocess](../history/20261018_E10_consolidation/experiments/E8_fast_robust_postprocess/README.md) | E8：快速、稳定的高质量后处理与模型联调 |
| [E9_model_postprocess_alignment](../history/20261018_E10_consolidation/experiments/E9_model_postprocess_alignment/README.md) | E9：面向 E8 后处理的模型改进 |

## 20261020_E25_consolidation

| 实验 | 原标题 |
|---|---|
| [E10_integrated_baseline](../history/20261020_E25_consolidation/experiments/E10_integrated_baseline/README.md) | E10：E7 模型与 E8 后处理联合基线 |
| [E11_short_budget_diagnosis](../history/20261020_E25_consolidation/experiments/E11_short_budget_diagnosis/README.md) | E11：短预算下草图与结构的作用诊断 |
| [E12_distillation_readiness](../history/20261020_E25_consolidation/experiments/E12_distillation_readiness/README.md) | E12：教师—学生表示—轻量投影的蒸馏准备 |
| [E13_solver_distillation](../history/20261020_E25_consolidation/experiments/E13_solver_distillation/README.md) | E13：几何与结构蒸馏 |
| [E14_stable_teacher](../history/20261020_E25_consolidation/experiments/E14_stable_teacher/README.md) | E14：MIB 修正与尾部稳定性 |
| [E15_quality_curriculum](../history/20261020_E25_consolidation/experiments/E15_quality_curriculum/README.md) | E15：从头训练与质量目标渐进对照 |
| [E16_reference_curriculum](../history/20261020_E25_consolidation/experiments/E16_reference_curriculum/README.md) | E16：原参考监督与质量目标渐进对照 |
| [E17_e10_geometry_v5](../history/20261020_E25_consolidation/experiments/E17_e10_geometry_v5/README.md) | E17：E10几何训练方法在当前v5数据上的对照 |
| [E18_fast_postprocess](../history/20261020_E25_consolidation/experiments/E18_fast_postprocess/README.md) | E18：固定E17模型的快速后处理 |
| [E19_mib_consistent_geometry](../history/20261020_E25_consolidation/experiments/E19_mib_consistent_geometry/README.md) | E19：训练与部署一致的MIB形状解码 |
| [E20_mib_fast_pipeline](../history/20261020_E25_consolidation/experiments/E20_mib_fast_pipeline/README.md) | E20：MIB一致性模型与快速后处理联合流程 |
| [E21_original_model_fast](../history/20261020_E25_consolidation/experiments/E21_original_model_fast/README.md) | E21：保留原E10模型的同版快速后处理对照 |
| [E22_native_geometry](../history/20261020_E25_consolidation/experiments/E22_native_geometry/README.md) | E22：剩余Python接触运算的原生实现 |
| [E24_reference_fast](../history/20261020_E25_consolidation/experiments/E24_reference_fast/README.md) | E24：冻结E17与原生快速路径的直接对照 |

## 20261021_E29_consolidation

| 实验 | 原标题 |
|---|---|
| [E23_capacity_geometry](../history/20261021_E29_consolidation/experiments/E23_capacity_geometry/README.md) | E23：持续参考监督下的容量对照 |
| [E26_tail_constraints](../history/20261021_E29_consolidation/experiments/E26_tail_constraints/README.md) | E26：尾部软约束的协调修复 |
| [E27_overlap_weight](../history/20261021_E29_consolidation/experiments/E27_overlap_weight/README.md) | E27：单独提高重叠穿透惩罚 |
| [E28_iterative_draft](../history/20261021_E29_consolidation/experiments/E28_iterative_draft/README.md) | E28：保留端点信息的学习式迭代草稿 |

## 20261026_flash_pro_consolidation

| 实验 | 原标题 |
|---|---|
| [baseline](../history/20261026_flash_pro_consolidation/experiments/baseline/README.md) | 无训练硬约束基线 |
| [E25_integrated_baseline](../history/20261026_flash_pro_consolidation/experiments/E25_integrated_baseline/README.md) | E25：冻结整合参照 |
| [E30_stable_fast_geometry](../history/20261026_flash_pro_consolidation/experiments/E30_stable_fast_geometry/README.md) | E30：两线程下的质量、速度与尾部稳定性 |
| [E31_conic_workspace_reuse](../history/20261026_flash_pro_consolidation/experiments/E31_conic_workspace_reuse/README.md) | E31：锥求解器工作区复用的可行性 |
| [E32_learned_relation_feedback](../history/20261026_flash_pro_consolidation/experiments/E32_learned_relation_feedback/README.md) | E32：学习关系决策与锥求解器反馈 |
| [E33_learned_solver_portfolio](../history/20261026_flash_pro_consolidation/experiments/E33_learned_solver_portfolio/README.md) | E33：从训练反事实中学习求解策略选择 |
| [E34_sparse_contact_feedback](../history/20261026_flash_pro_consolidation/experiments/E34_sparse_contact_feedback/README.md) | E34：相同关系权重的稀疏接触推理 |
| [E35_optimizer_attribution](../history/20261026_flash_pro_consolidation/experiments/E35_optimizer_attribution/README.md) | E35：E25优化器的触发与贡献归因 |
| [E36_persistent_conflict_repair](../history/20261026_flash_pro_consolidation/experiments/E36_persistent_conflict_repair/README.md) | E36：持续冲突的局部修复与阶段控制 |
| [E37_radial_draft_expansion](../history/20261026_flash_pro_consolidation/experiments/E37_radial_draft_expansion/README.md) | E37：draft位置径向扩展5%的受控研究 |
| [E38_model_architecture](../history/20261026_flash_pro_consolidation/experiments/E38_model_architecture/README.md) | E38：面向高质量draft的模型容量与结构研究 |
| [E39_rectangular_attention](../history/20261026_flash_pro_consolidation/experiments/E39_rectangular_attention/README.md) | E39：矩形感知注意力 |
| [E40_endpoint_constraint_tokens](../history/20261026_flash_pro_consolidation/experiments/E40_endpoint_constraint_tokens/README.md) | E40：外部端点与约束对象的分层交互 |
| [E41_rectangular_depth](../history/20261026_flash_pro_consolidation/experiments/E41_rectangular_depth/README.md) | E41：E39矩形注意力继续扩深 |
| [E43_overlap_area_loss](../history/20261026_flash_pro_consolidation/experiments/E43_overlap_area_loss/README.md) | E43：联合正则化矩形96×12，新增重叠面积训练损失 |
| [E44_near_limit_proposals](../history/20261026_flash_pro_consolidation/experiments/E44_near_limit_proposals/README.md) | 接近加权 Q=1.01 的结构生成研究（当前单例预算 2 秒） |
| [E45_draft_fidelity](../history/20261026_flash_pro_consolidation/experiments/E45_draft_fidelity/README.md) | E45：持续尊重原始 draft 的后处理 |
| [E46_global_draft_refinement](../history/20261026_flash_pro_consolidation/experiments/E46_global_draft_refinement/README.md) | E46：全局布局保持与质量/速度联合优化 |
| [E47_draft_topology_confidence](../history/20261026_flash_pro_consolidation/experiments/E47_draft_topology_confidence/README.md) | E47：保留完整draft的全局关系与置信度研究 |
| [E48_quality_potential](../history/20261026_flash_pro_consolidation/experiments/E48_quality_potential/README.md) | E48：5秒质量优先的draft潜力评估求解器 |
| [E49_monotone_quality](../history/20261026_flash_pro_consolidation/experiments/E49_monotone_quality/README.md) | E49：逐例质量保护与5秒后处理优化 |
| [E51_module_overlap](../history/20261026_flash_pro_consolidation/experiments/E51_module_overlap/README.md) | E51：逐模块相对重叠、面积重要性与连续个数因子 |
| [E52_training_throughput](../history/20261026_flash_pro_consolidation/experiments/E52_training_throughput/README.md) | E52：相同模型与统一Loss的训练吞吐优化 |
| [E53_loss_geometry](../history/20261026_flash_pro_consolidation/experiments/E53_loss_geometry/README.md) | E53：冻结模型与flash，只研究固定Loss |
| [E54_reference_extremes](../history/20261026_flash_pro_consolidation/experiments/E54_reference_extremes/README.md) | E54：不参考reference／只参考reference |

## 20261029_model_research_consolidation

| 实验 | 原标题 |
|---|---|
| [E55_reference_position_shape](../history/20261029_model_research_consolidation/experiments/E55_reference_position_shape/README.md) | E55：只拟合 reference 的位置与形状 |
| [E56_e55_geometry_finetune](../history/20261029_model_research_consolidation/experiments/E56_e55_geometry_finetune/README.md) | E56：E55 best40 的 E42 Loss 后训练 |
| [E57_capacity_data_diagnosis](../history/20261029_model_research_consolidation/experiments/E57_capacity_data_diagnosis/README.md) | E57：当前架构表达能力与训练规模诊断 |
| [E58_training_data_v6](../history/20261029_model_research_consolidation/experiments/E58_training_data_v6/README.md) | E58：v6训练数据扩展至64k与缓存准备 |
| [E59_centerline_overlap](../history/20261029_model_research_consolidation/experiments/E59_centerline_overlap/README.md) | E59：中心连线分离损失与扩大训练 |
| [E60_scaling_diagnosis](../history/20261029_model_research_consolidation/experiments/E60_scaling_diagnosis/README.md) | E60：扩大训练规模与归一化的诊断 |
| [E61_explosion_replay](../history/20261029_model_research_consolidation/experiments/E61_explosion_replay/README.md) | E61：爆炸首次更新复现与无图聚合前归一化重训 |
| [E62_e42_loss_update_audit](../history/20261029_model_research_consolidation/experiments/E62_e42_loss_update_audit/README.md) | E62：E42实际Adam更新与Loss阻力检查 |
| [E63_e42_center_relaxation](../history/20261029_model_research_consolidation/experiments/E63_e42_center_relaxation/README.md) | E63：E42 温和放宽中心约束后训练 |
| [E64_e42_unified_retrain](../history/20261029_model_research_consolidation/experiments/E64_e42_unified_retrain/README.md) | E64：E42统一固定Loss从头重训 |
| [E65_geometry_messages](../history/20261029_model_research_consolidation/experiments/E65_geometry_messages/README.md) | E65：冻结G下的几何消息与匹配容量对照 |
| [E66_shared_refinement](../history/20261029_model_research_consolidation/experiments/E66_shared_refinement/README.md) | E66：冻结G下的增量输出与共享迭代修正 |
| [E67_multiscale_tokens](../history/20261029_model_research_consolidation/experiments/E67_multiscale_tokens/README.md) | E67：冻结G下的多尺度分组与全局对照 |
| [E68_port_tokens](../history/20261029_model_research_consolidation/experiments/E68_port_tokens/README.md) | E68 v2：完整端口组合方案 |
| [E70_reference_task_loss](../history/20261029_model_research_consolidation/experiments/E70_reference_task_loss/README.md) | E70：新任务Loss＋完整端口 |

## 20261101_research_consolidation

| 实验 | 原标题 |
|---|---|
| [E69_case_loss_statistics](../history/20261101_research_consolidation/experiments/E69_case_loss_statistics/README.md) | E69：不同案例的Loss差异与草稿质量统计 |
| [E71_density_reference_curriculum](../history/20261101_research_consolidation/experiments/E71_density_reference_curriculum/README.md) | E71：温和reference日程＋内部密度＋紧凑约束 |
| [E72_topology_posttrain](../history/20261101_research_consolidation/experiments/E72_topology_posttrain/README.md) | E72：扩大模型＋近邻拓扑监督后训练 |
| [E73_wide_robust_posttrain](../history/20261101_research_consolidation/experiments/E73_wide_robust_posttrain/README.md) | E73：保留图聚合映射的宽模型后训练 |
| [E75_collision_box_finetune](../history/20261101_research_consolidation/experiments/E75_collision_box_finetune/README.md) | E75：E74候选上的真实交面积与外框后训练 |
| [E76_bounded_refinement](../history/20261101_research_consolidation/experiments/E76_bounded_refinement/README.md) | E76：冻结E74主干，只学习有界局部修正 |
| [E77_compact_refinement](../history/20261101_research_consolidation/experiments/E77_compact_refinement/README.md) | E77：固定E74预测外框，学习内部重排 |
| [E78_connectivity_posttrain](../history/20261101_research_consolidation/experiments/E78_connectivity_posttrain/README.md) | E78：同形组协调输出与固定模块通道后训练 |
| [E79_center_refinement](../history/20261101_research_consolidation/experiments/E79_center_refinement/README.md) | E79：保留E78同形尺寸，单独学习中心修正 |
| [E80_modular_transfer](../history/20261101_research_consolidation/experiments/E80_modular_transfer/README.md) | E80：冻结修正模型，替换布局模型 |
| [E82_global_pressure_refinement](../history/20261101_research_consolidation/experiments/E82_global_pressure_refinement/README.md) | E82：冻结修正模型，补充全局公共通道信息 |
| [E83_strategy_attribution](../history/20261101_research_consolidation/experiments/E83_strategy_attribution/README.md) | E83：先用两个零训练对照拆解策略贡献 |
| [E84_loss_control](../history/20261101_research_consolidation/experiments/E84_loss_control/README.md) | E84：同一初始化、同一模型，单独检验新增Loss目标包 |
| [E86_independent_global_corrector](../history/20261101_research_consolidation/experiments/E86_independent_global_corrector/README.md) | E86：独立全局神经修正网络 |
| [E87_simple_supervised_placer](../history/20261101_research_consolidation/experiments/E87_simple_supervised_placer/README.md) | E87：监督布局器的简化Loss与稀疏数据研究方案 |
| [E88_sparse_training](../history/20261101_research_consolidation/experiments/E88_sparse_training/README.md) | E88：第二优先级，真实稀疏样本覆盖 |
| [E89_spatial_neural_refinement](../history/20261101_research_consolidation/experiments/E89_spatial_neural_refinement/README.md) | E89：第三优先级，二维神经空间模块 |
| [E91_e74_32k_scratch](../history/20261101_research_consolidation/experiments/E91_e74_32k_scratch/README.md) | E91：E74的32k数据和192×12结构，使用新Loss从头训练 |
| [E92_two_stage_placer](../history/20261101_research_consolidation/experiments/E92_two_stage_placer/README.md) | E92：布局网络前训练／后训练规范 v1 |
| [E93_matched_posttrain](../history/20261101_research_consolidation/experiments/E93_matched_posttrain/README.md) | E93：P best36共同起点的匹配后训练 |
| [E94_posttrain_lr_reset](../history/20261101_research_consolidation/experiments/E94_posttrain_lr_reset/README.md) | E94：后训练学习率重置到1e-3 |

## 20261103_final_delivery_consolidation

| 实验 | 原标题 |
|---|---|
| [E42_joint_regularization](../history/20261103_final_delivery_consolidation/experiments/E42_joint_regularization/README.md) | E42：12层矩形注意力的联合正则化AB |
| [E50_anytime_quality](../history/20261103_final_delivery_consolidation/experiments/E50_anytime_quality/README.md) | E50：并行保护轨迹与约束反馈驱动的随时可用求解器 |
| [E74_collision_32k_posttrain](../history/20261103_final_delivery_consolidation/experiments/E74_collision_32k_posttrain/README.md) | E74：真实碰撞门控Loss＋冻结嵌套32k后训练 |
| [E81_multisource_refinement](../history/20261103_final_delivery_consolidation/experiments/E81_multisource_refinement/README.md) | E81：小布局模型＋暖启动加宽的独立修正模型 |
| [E85_cooperative_space_refinement](../history/20261103_final_delivery_consolidation/experiments/E85_cooperative_space_refinement/README.md) | E85：E74上的协同空间修正 |
| [E90_auxiliary_weight_placer](../history/20261103_final_delivery_consolidation/experiments/E90_auxiliary_weight_placer/README.md) | E90：提高E87辅助权重的固定Loss对照 |
| [E95_single_pass_corrector](../history/20261103_final_delivery_consolidation/experiments/E95_single_pass_corrector/README.md) | E95：独立全局修正器 |
| [E96_port_compression](../history/20261103_final_delivery_consolidation/experiments/E96_port_compression/README.md) | E96：完全压缩的独立全局修正器 |
| [E97_gpu_hybrid_solver](../history/20261103_final_delivery_consolidation/experiments/E97_gpu_hybrid_solver/README.md) | E97：GPU混合神经网络与几何求解 |
| [E99_attraction_repulsion](../history/20261103_final_delivery_consolidation/experiments/E99_attraction_repulsion/README.md) | E99：有界矩形中心吸引—排斥 |
| [E100_relation_guided_placer](../history/20261103_final_delivery_consolidation/experiments/E100_relation_guided_placer/README.md) | E100：关系引导的同时布局网络 |
