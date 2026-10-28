# 2026-10-29历史实验归档

用户授权整理工作区并继续自主模型研究。11个已结束或已停止的实验移入本目录，共4,528个文件。未删除checkpoint、缓存、源码或唯一运行证据；同盘目录迁移后逐文件核对存在、字节数和UTC修改时间，见[migration.json](migration.json)。历史JSON、日志、命令、冻结索引和源码均不因迁移改写。

日常工作区保留E42 draft/flash、E50 pro、E61稳定大模型、E65几何消息候选及E68/E69/E70/E71最近四项研究。新研究E72是独立复制实现，不恢复旧控制器、旧自动唤醒或未完成的退役路线。

| 历史实验 | 保留原因 |
|---|---|
| [E55](experiments/E55_reference_position_shape/README.md)、[E56](experiments/E56_e55_geometry_finetune/README.md) | 仅reference监督失败及其后训练取舍 |
| [E57](experiments/E57_capacity_data_diagnosis/README.md)、[E58](experiments/E58_training_data_v6/README.md) | 容量/数据规模诊断、已冻结16k/32k/64k嵌套数据及缓存生产证据 |
| [E59](experiments/E59_centerline_overlap/README.md)、[E60](experiments/E60_scaling_diagnosis/README.md) | 归一化代价、失稳与拟合诊断 |
| [E62](experiments/E62_e42_loss_update_audit/README.md)、[E63](experiments/E63_e42_center_relaxation/README.md) | 梯度冲突证据、E42后训练control和失败的中心放宽 |
| [E64](experiments/E64_e42_unified_retrain/README.md) | 固定G从头重训协议参照 |
| [E66](experiments/E66_shared_refinement/README.md)、[E67](experiments/E67_multiscale_tokens/README.md) | 已失败/用户停止的迭代与token路线，保留唯一断点 |

旧路径`experiments/<名称>/...`统一映射为`history/20261029_model_research_consolidation/experiments/<名称>/...`。历史命令依赖原始目录或相对根定位，不能直接把迁移后的代码当成新实验运行；如需严格复现，先按manifest恢复所需目录到原位置，检查没有同名目录和活跃任务，再执行原入口。新实验直接读取归档结果时使用迁移后的路径，不关闭身份检查。

E58扩展缓存随生产者归档；其manifest内原路径保持冻结。E61已结束的旧复现入口依赖该原路径，因此未来需要扩展数据时，必须为新实验实现明确的只读路径映射并验证原manifest，或按上段恢复生产者。E72当前仅使用E42原位的v5共享缓存，不受此迁移影响。保留候选的旧源码与配置仍冻结，不将维护性迁移伪称原始运行。

当前用户新授权见[E72方案](../../experiments/E72_topology_posttrain/README.md)。封存测试未读取。
