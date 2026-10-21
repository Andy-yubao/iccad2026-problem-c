# 当前最佳E25与未运行E29整理

2026-10-22后续变更：用户已要求删除原E29方案，其实验目录、专属目标与计划已移除，编号29可供后续新实验复用。本归档目录名及迁移清单保留发生时身份，不代表仍有待运行E29，也不保留其方案副本。以下描述2026-10-21的历史整理事实。

2026-10-21，按用户要求保存两版：整理前的完整实验记录已提交为`97ba1eed70d4027e121a95466037661e2aa7bcc3`，并核验GitHub origin/main后才迁移；第二版保留当前最佳组合和未运行E29。

整理发生时，日常实验区为[E25](../../experiments/E25_integrated_baseline/README.md)、当时未运行的E29及按核心约定保留的[无训练baseline](../../experiments/baseline/README.md)。E25最佳模型、数值求解代码、共享缓存及既有结果保持不变；当时E29仅目标定义，无模型实现、训练或新机制检查。

本次归档的实验：

| 实验 | 已有结论 |
|---|---|
| [E23容量对照](experiments/E23_capacity_geometry/README.md) | 训练拟合改善，开发/最终质量未优于保留基线 |
| [E26尾部修复](experiments/E26_tail_constraints/README.md) | 用户要求暂停；保留既有五例及图集证据，不恢复研究 |
| [E27重叠权重](experiments/E27_overlap_weight/README.md) | 冻结best69；同版后处理加权Q1.061580，未替换E25 |
| [E28学习式迭代草稿](experiments/E28_iterative_draft/README.md) | best48，原始dev500/dev2000零硬合法，其他三组未跑 |

E27另有用户已完成的[steady后处理500例结果](experiments/E27_overlap_weight/runs/20261021_e27_single_steady_monitor500/results.json)：500/500硬合法、加权Q1.052754685、平均0.279705909秒，仍未替换E25。此处只摘录已有summary，不重评分、不追加比较或推理。

精确旧新路径、字节数、mtime及必要SHA256见[manifest.json](manifest.json)。可重建张量缓存只核对大小和mtime，同盘移动而不全量重哈希；唯一权重、正式结果和源码按SHA256核对。缓存、权重及Python缓存保持本机原样，没有删除唯一证据，也不上传GitHub。与本次实验整理无关的`output/`未改动、未入库。

恢复时先查清单，再将需要的归档目录同盘移回原路径；确认原路径空闲且没有活跃任务后再操作。历史README、JSON、日志、命令、源码快照中的旧路径保持原样，使用清单定位；原相对链接若依赖旧目录深度，不代表历史结果缺失。历史旧实验的导入/资源路径未改写，不把归档视为随时可直接续跑的当前入口。源码亦可从整理前提交恢复；Git不保存本机权重及数据缓存。

E25原`migration_check.py`依赖“E23尚未训练”的旧前提，移到[legacy_E25_migration_check.py](legacy_E25_migration_check.py)保留，不作为活跃入口。[迁移脚本](archive_workspace.py)和[静态验收](verification.json)记录本次整理；迁移脚本仅供审计，不应重复运行。

本次没有训练、推理、求解、重评分、读取封存测试或下载数据。更早历史仍见[20261020归档](../20261020_E25_consolidation/README.md)。
