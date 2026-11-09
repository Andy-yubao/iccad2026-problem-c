# 工作区结构

更新：2026-11-05。当前工作区按“规则 → 接手判断 → 实验原证据”阅读；不把历史归档当待运行队列。[全部实验实际位置](experiment_catalog.md) · [当前状态](current_status.md)。

| 位置 | 职责 |
|---|---|
| README.md / AGENTS.md | 总入口 / 简短现行边界 |
| docs/START_HERE.md | 接手模型的最小背景与判断入口 |
| docs/research_history.md | 按问题组织的研究结论、失败机制、限制和证据 |
| docs/current_status.md | 带日期的运行状态快照，不取代run内机器状态 |
| docs/experiment_catalog.md | 按实际目录生成的实验索引，不推断完成状态 |
| docs/protocol.md / datasets.md / gpu_training.md / git.md | 可复用协议、数据、工程、源码规则 |
| experiments/En_xxx/ | E98、E101–E116源码、计划、来源和runs；E116原四层→新版首稿按默认16k获授权后台串行训练；E114/E115历史36k身份保留 |
| experiments/En_xxx/runs/ | 冻结身份、源码快照、状态、日志、best/latest、坐标、报告 |
| datasets/ / scoring/ / visualization/ | 共享数据读取、官方评分与独立硬检查、纯展示 |
| protocol/ | 原始split、冻结成员、面板及paths.json；旧索引不改；random_reference40k_v1父集合与random_reference16k_v1训练视图、研究规则 |
| references/ / third_party/ | 官方材料、阅读资料及固定上游代码 |
| history/ | 退役实验和旧文档原件、迁移清单；[归档导航](../history/README.md) |
| output/ | 面向人的交付与研究提案；[入口](../output/README.md)，提案不等于已实施 |
| .conda/ / .cache/ | 独立环境、有效缓存；目录名含cache不代表可删 |
| tmp/ | 临时工作位置；一次性准备/诊断脚本已归本次history，不作运行入口 |

## 受保护的路径和依赖

- E101 `assets/c5_baseline.pt` 是原C5；`delivery/model.pt`是首次64k暖启动导出；历史链路导出在 `runs/20261103_replay/sf/experiments/E101_final_c5_solver/delivery/model.pt`。不能按同名model.pt替换。
- E101 replay和chain64k的`s42/s74/sf`为隔离运行根，含共享数据junction/缓存硬链接；本次不移动、不递归清理这些目录。
- E42原有效缓存位于 `history/20261103_final_delivery_consolidation/experiments/E42_joint_regularization`；E58扩展缓存位于 `history/20261029_model_research_consolidation/experiments/E58_training_data_v6`。生产者归档不代表缓存失效，消费者按已登记映射读取。
- E103 assets/sources/evidence是五个来源模型的保护副本；E106独立保存三个来源推理模型与25000标签。是否“重复”不能只按文件名或内容判断，因为它们承载冻结身份。
- 每个实验的best/latest、唯一导出、失败现场、STOP、有效缓存、原始数据和环境均保护；旧STOP不清除。

## 历史位置的解释

20261103归档收纳E42/E50/E74/E81/E85/E90/E95/E96/E97/E99/E100等，E98后来整体迁回experiments，映射见该归档的`E98_restore_20261104.json`。E55–E70等主要位于20261029归档，E71–E94部分位于20261101归档；准确路径以实验目录索引为准。

历史README、JSON和命令保留发生时的路径与授权。旧链接可能依赖原工作区，先查对应manifest；不为“修链接”批改冻结源码或运行证据。严格复现须登记新入口/读取适配或明确恢复原目录，不能绕过身份检查。

## 本次整理

原16份文档（含后补结构提案快照）和Git状态已保存到 [20261105_workspace_handoff](../history/20261105_workspace_handoff/README.md)。tmp中的一次性准备/诊断脚本按哈希迁入该归档；现行实验源码、所有runs和数值资产原位保留。本次目标是减少过期导航和临时脚本干扰，不以释放大体积磁盘为目的。

后续清理顺序：进程 → 引用/路径 → 原件快照 → 同盘迁移 → 清单与哈希核对 → 当前导航。移动/递归删除前核实绝对目标在指定工作区内。Git初始大量删除/未跟踪来自既有整理，不能据此reset、批量暂存或擅自提交。
