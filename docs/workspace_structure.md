# 工作区结构

| 位置 | 职责 |
|---|---|
| `README.md` / `AGENTS.md` | 项目入口 / 简短目标与导航 |
| `docs/` | 当前协议、结果摘要、研究结论、数据与工程说明 |
| `protocol/` | 冻结数据成员、面板、模型身份、机器协议与路径 |
| `experiments/` | E119/E120及后续E131–E133；各自维护源码、计划、状态和runs |
| `datasets/` | 原始数据、已生成NPZ与持久化CPU张量缓存 |
| `scoring/` | 官方评分适配、独立硬检查、几何诊断 |
| `visualization/` | 唯一已存布局展示与PNG导出；数据接口见其README |
| `history/` | 退役实验、旧文档及清理清单；不作待执行队列 |
| `references/` / `third_party/` | 官方材料、阅读资料、固定上游源码 |
| `.conda/` / `.cache/` | 环境、有效运行缓存及可视化服务日志 |
| `tmp/` | 一次性工作文件，不作长期运行入口 |

长期结果归所属实验的 `runs/` 或独立 `evaluation_*/`；不要重新建立根 `output/` 作为杂物目录。实验内部已登记的同名 `output/` 是历史评价证据，本次没有删除。

2026-11-11的29个退役实验保留在 `history/20261111_experiments_consolidation/experiments/`；E101深层隔离运行根部分路径超过MAX_PATH，读取用Windows长路径。2026-11-13旧可视化与旧文档移入 `history/20261113_workspace_cleanup/`，现行只保留一套样式。

保护权重、best/latest、失败现场、STOP、数据原件和有效缓存。整理前检查实际运行、依赖、链接与绝对路径；旧实验按原身份复现，不批改冻结源码/日志里的历史路径。恢复归档或旧STOP需明确授权。Git有既有迁移改动，不reset或批量暂存，见 [Git约定](git.md)。
