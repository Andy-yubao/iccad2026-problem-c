# 实验索引

| 目录 | 当前用途 |
|---|---|
| [E24_reference_fast](E24_reference_fast/README.md) | 用户已停止研究；三个500例对照完成，保留steady：硬合法500/500、加权Q1.039578、平均0.259487秒，尚未同时达标 |
| [E23_capacity_geometry](E23_capacity_geometry/README.md) | 仅完成宽度160/1,370,771参数的五例检查，正式训练未启动；用户已停止研究 |
| [E22_native_geometry](E22_native_geometry/README.md) | 原生接触运算五例逐值完全一致，冷JIT单列；完整500硬合法但Q1.074412/0.284301秒未达标，函数级加速未转化为整流程成功 |
| [E21_original_model_fast](E21_original_model_fast/README.md) | 已准备原保留E10模型的同版快速后处理对照，五例逐值验证实现等价；不续训历史任务，待E20结果后决定评价 |
| [E20_mib_fast_pipeline](E20_mib_fast_pipeline/README.md) | 已准备联合流程：等待E19完成后冻结唯一best，独立复用E18快速几何算法，增加无解时廉价LP前门控；正式时间比较待训练/主评价结束 |
| [E19_mib_consistent_geometry](E19_mib_consistent_geometry/README.md) | 已完成62轮/best54：500硬合法、加权Q1.075417/平均1.999252秒，同版后处理较E17退化；冻结唯一best交E20快算法对照，不续训或换checkpoint |
| [E18_fast_postprocess](E18_fast_postprocess/README.md) | 用户授权长期自主研究：固定E17模型，优先研究快速后处理以同时达到加权Q<1.04与平均单例<0.25s；开发期测试封存 |
| [E17_e10_geometry_v5](E17_e10_geometry_v5/README.md) | 已完成：第56轮早停，best51；固定500例硬合法500/500、加权Q1.060518、平均1.915333s，作为E18固定模型参照 |
| [E16_reference_curriculum](E16_reference_curriculum/README.md) | 原三组已完成；新增497,299参数的第16/21轮衰减对照，复用v5缓存，用户授权后台依次训练、每组最多4h；本轮不启动后处理 |
| [E15_quality_curriculum](E15_quality_curriculum/README.md) | v4三组对照：共享缓存、训练验证零后处理、dev500代理选模；第一组定稿best66轮，CPU终末500例评价可与后两组GPU训练并行，后两组三档学习率；每组仅一次终末后处理 |
| [E14_stable_teacher](E14_stable_teacher/README.md) | 当前修正任务流程：100＋另500验收通过，600硬合法、无Q>1.8/退化；500均值1.039376/最大1.458786，附额外计算成本；新版16k/2k/2k规范、测试封存和教师生成入口 |
| [E10_integrated_baseline](E10_integrated_baseline/README.md) | 当前联合基线：E7冻结模型＋E8 conic_robust；MIB平均面积派生dev100配对诊断完成：平均Q=1.046346、硬合法100/100、MIB零违反100/100，非原始任务成绩 |
| [E11_short_budget_diagnosis](E11_short_budget_diagnosis/README.md) | dev100结构/解码/数值接纳复核完成；仅关系不足，整体几何oracle配实验修正版接近目标但未达标；E10未改、未训练 |
| [E12_distillation_readiness](E12_distillation_readiness/README.md) | 两组开发100例理想/2%几何噪声接口通过预注册门槛；修复+通信0.13～0.14秒，教师结构仍为oracle；训练标签pilot与可恢复生成入口，无学生训练成绩 |
| [E13_solver_distillation](E13_solver_distillation/README.md) | E12接口的几何/几何+结构蒸馏实施；完整教师库、可恢复训练、真实学生评价启动脚本已交付，五例实现检查通过，正式长任务未启动 |
| [baseline](baseline/README.md) | 保留的无训练硬约束对照与官方模板诊断 |

其他实验退出日常工作区入口，原始证据见[2026-10-18归档](../history/20261018_E10_consolidation/README.md)。归档不代表获准恢复训练、测试或废弃路线。新增研究须明确假设、对照、面板、预算及停止条件，使用 `En_xxx` 命名。
