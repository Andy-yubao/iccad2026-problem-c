# 数据版本与位置

数据由[protocol/paths.json](../protocol/paths.json)解析，官方原件和[安装清单](../datasets/datasets_manifest.json)保留，不重复下载/解压。原训练约9000分片；官方公开验证100例仅作兼容和公开对照。

| 索引 | 作用 |
|---|---|
| split_v1.json | 原始完整分片80/10/10；7200/900/900分片，始终冻结 |
| workset_v2.json | 历史20k/2500/2500任务，不再作新训练默认 |
| derived_workset_v3.json | 公开MIB平均面积派生任务16k/2k/2k |
| distillation_workset_v4.json | 历史教师选优分布，只用于对应实验 |
| reference_workset_v5.json | 当前reference任务16k/2k/2k，固定train/dev500 |
| reference_workset_v6.json | 嵌套扩展集合；E95读取经核验的原16k前缀 |

索引均在protocol/，不覆盖或重新抽样。扩展训练须登记实际成员及生成器历史曝光。MIB平均面积仅依公开输入修改派生任务，原数据不变；不能把任务变化当成模型提升。

原始训练为datasets/floorset_lite/worker_*/layouts_*.th，公开验证为datasets/LiteTensorDataTest/，压缩原件为datasets/archives/。实际位置始终以paths.json为准。E101使用已验证E58的64k嵌套成员与原E42有效缓存，转换为相同FP32磁盘映射供按批传输。E58仍在原历史目录，E42缓存现位于history/20261103_final_delivery_consolidation/experiments/E42_joint_regularization，由E101路径映射解析。旧缓存均保护，不是待删临时文件。

训练仅用冻结train白名单，dev不反传。测试默认封存，旧v2已暴露957例/39个worker的历史不改写；v5选择已排除相关暴露。新测试须用户明确发起。公开固定/预放置规格可输入，自由reference只作监督/展示/计分。

共享datasets处理路径、成员读取、面积变换与标签遮罩；图特征/采样/Loss归实验。训练原标签WHXY须转换成输出XYWH，公开验证多边形不能套用同字段。缺失时先核实路径和安装证据，再按scripts/download_datasets.ps1恢复，日常不运行重解压。
