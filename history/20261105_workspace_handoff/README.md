# 2026-11-05 工作区与模型接手整理

目的：让新接手模型快速了解已经尝试过什么、哪些判断有支持/反例、哪些方向只是提案。本文记录整理操作，不产生研究启动授权。

[完整改动与保护清单](manifest.json)。

## 交付入口

- [接手导读](../../docs/START_HERE.md)：最小任务背景、八条关键判断、最短原证据阅读链。
- [研究综合](../../docs/research_history.md)：按问题整理的历史、对照、失败机制、数表及未决问题。
- [当前状态](../../docs/current_status.md)：区分运行状态、准备快照与后续授权。
- [全部实验目录](../../docs/experiment_catalog.md)：106个实际实验根（含不同归档阶段/非编号原型），不将目录数当独立完整训练数。
- [README](../../README.md)、[协议](../../docs/protocol.md)、[结构](../../docs/workspace_structure.md)、[AGENTS](../../AGENTS.md)：清除过期队列式叙述，保留现行边界；各近期实验README同步完成/停止状态。

## 实际清理与保护

整理前保存15份入口/协议/实验README；后来给结构提案补当前状态说明前再保存其原件，共16份，见[documents_snapshot.json](documents_snapshot.json)。每份快照有原路径、快照路径、字节和SHA256；旧版本不删除。

将tmp中11份已完成的一次性准备/诊断脚本（共52726字节）移到one_off_scripts；逐份保持size/mtime并核对SHA256。[迁移清单](moves.json)。当前数值源码入口未引用这些tmp路径；归档脚本保留发生时的路径，只供审计，不直接重跑。tmp现为空。

没有移动现行实验目录或任何runs，未删除数据、环境、有效缓存、权重、STOP或失败现场，没有以“重复”名义删除冻结副本；没有训练、模型推理、后处理求解、重评分、读取测试载荷或改变自动任务。归档是减少导航噪声，不声称释放大体积磁盘。

## 核验与局限

- [进程快照](process_snapshot.json)：北京时间15:58:51未发现python/pythonw/conda；CIM不可用，使用Get-Process名称核查，不是持续监控。
- [运行状态原件摘录](run_state_snapshot.json)：18份小型JSON，保留源路径/哈希；不重写原run。
- [实验根清单](experiment_catalog.json)：按实际目录建立，排除运行隔离根与文档副本。
- [文档/链接/迁移检查](verification.json)：只核对这次现行导航与引用证据，不声称修复全部历史失效链接或重做科学审计。
- [引用证据清单](evidence_index.json)：本次导读/综合引用原件的字节与SHA256，便于后续识别漂移；不读取数值数据或权重。
- [整理前Git状态](git_status_before.txt)及[计数](git_summary_before.json)在改入口文档前保存（当时本归档已创建）。既有大量删除/未跟踪源于此前整理，未reset、批量暂存、commit或push。

今后恢复历史先查各归档manifest，不批改冻结源码/JSON里的旧路径，不把旧README中的“正在运行/启动/预算”当现行指令。原始证据和本次综合解释分层保存，后续模型可独立质疑本文的判断。
