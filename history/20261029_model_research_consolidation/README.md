# 2026-10-29历史实验归档

用户授权整理工作区并继续自主模型研究。分两阶段共15个已结束或已停止的实验移入本目录，原迁移清单合计5,746个文件。未删除checkpoint、训练张量缓存、源码或唯一运行证据；首次迁移11个实验见[migration.json](migration.json)，E74阶段结束后迁移E61/E65/E68/E70见[migration_after_E74.json](migration_after_E74.json)。随后仅清除72个可重建CPython/Numba编译缓存，见[compiled_cache_cleanup.json](compiled_cache_cleanup.json)；保留的5,674个文件逐一核对大小和UTC时间不变，缺失项严格等于已登记的72个缓存，见[final_integrity.json](final_integrity.json)。历史JSON、日志、命令、冻结索引和源码均不因迁移改写。

日常工作区保留E42 draft/flash、E50 pro、E69通用诊断、E71紧凑草稿对照和E72至E75近期研究，其中E74为当前候选、E75为进行中的后训练。E61/E65/E68/E70已结束且无活跃控制器，随第二阶段归档；后续新实验独立复制实现，不恢复旧控制器、旧自动唤醒或未完成的退役路线。

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

第二阶段新增：[E61稳定性与大模型](experiments/E61_explosion_replay/README.md)、[E65几何消息](experiments/E65_geometry_messages/README.md)、[E68完整端口](experiments/E68_port_tokens/README.md)、[E70避碰与外框反例](experiments/E70_reference_task_loss/README.md)。其旧运行和源码字节保持原样；如需旧入口严格复现，仍须按原路径恢复。
