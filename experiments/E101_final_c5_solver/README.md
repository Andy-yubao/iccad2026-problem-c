# E101：原C5交付与重训证据

更新：2026-11-05。保留竞赛交付方法为原E97第二轮C5＋原完整单线程后处理。**本目录有多份不同权重：原C5、首次64k暖启动、历史链路重跑；另一个三阶段64k链路已停止。没有当前待自动启动的链路。**

## 身份与结果

| 对象 | 权重SHA256 | dev500加权Q | 已登记test2000加权Q | 状态 |
|---|---|---:|---:|---|
| 原历史C5，assets/c5_baseline.pt | 028ea3aa124c8cd9e7c3cc7bf8587c699ab9b4932f256899e3780d4d8c8816e0 | 1.012685893 | 1.018808539 | 原dev只读复用，2000补测完成 |
| 首次64k暖启动，delivery/model.pt | c1f0ce288bfe8fee350e0ee92b29f9523487eb61753b741d71c18fafe350f04e | 1.031581961 | 1.036082419 | L best31 / F best12，完成 |
| 历史链路重跑，replay内sf/delivery/model.pt | ca0465e48da7a0e1fc345aee28872e412209ff268edc774b43347c7496748d1a | 1.018307314 | 1.025842192 | E42 best51→E74 best43→F best6，完成 |
| 三阶段均64k，chain64k | 未导出完成组合 | 无 | 无 | E42 step63561用户停止，STOP保持 |

哈希从已冻结报告摘录，不因整理重新加载模型；准确资产位置和来源见各run。完成的2000评价均2000/2000硬合法、零回退。test2000在重训期间已曝光，原C5虽首次补测该权重，也不是全新未见面板。不得按这些Q事后重选模型/训练。

- 原C5：[2000报告](runs/20261104_original_C5_test2000/report.md) · [登记方案](runs/20261104_original_C5_test2000/README.md)。原历史dev完整均时204.464ms/P95 521.080ms；2000完整均时213.767ms/P95 530.940ms，540例超过250ms。
- 首次64k：[设计](DESIGN.md) · [冻结计划](plan.json) · [状态](runs/20261103_final64k_seed0/state.json) · [dev报告](runs/20261103_final64k_seed0/evaluation/dev_monitor500/report.md) · [2000报告](runs/20261103_final64k_seed0/evaluation/local_test/report.md)。它是原E74/C5权重暖启动，教师改为冻结C5，不是三阶段从头64k。
- 历史链路：[方案](runs/20261103_replay/README.md) · [状态](runs/20261103_replay/state.json) · [结果](runs/20261103_replay/report.md)。E42随机16k→E74后训练32k→原C3暖启动F16k，原E85教师；三个阶段均正常早停。隔离根`s42/s74/sf`和导出保护原位。
- 三阶段64k：[方案](runs/20261104_chain64k_seed0/README.md) · [停止核验](runs/20261104_chain64k_seed0/stop_verification_20261104.json)。原计划E42随机64k→E74 64k→F64k；实际后两阶段未启动，不自动恢复。

## 交付与评价边界

公开输入经每模块端口总权重/加权中心压缩，进入L、C5修正及内部几何，再由原完整CPU后处理和独立硬检查生成解。网络/求解接口不接收自由reference或评分分母。没有加入E99吸引排斥或E100关系头。原实现来源见[provenance.json](provenance.json)。

原C5仍未实现固定dev500 Q<1.01和逐例250ms，不能把低均时当尾部达标。完整逐例时间含公开输入构造/遮罩/面积变换、特征/传输、L/F/内部几何、全部后处理/回退、硬检查/同步；加载、预热、磁盘、评分/绘图另列。主线是公开MIB平均面积派生任务、中性时间本地Q，不称正式比赛成绩。

本目录原完整后处理与E98.depth_budget.v1不是同一评价口径，不能和E103–E105数字直接作无条件排名。研究解释见[历史综合](../../docs/research_history.md)。

推理API保留`experiments.E101_final_c5_solver.solver.Service`，按显式checkpoint初始化后solve(public_inputs)。本页不提供自动训练入口；历史命令只用于追溯，恢复/新实验按现行协议及用户授权登记。

首次64k的CUDA图/预算迁移、历史链路评分依赖修复等原件都在各run；详细整理前首页在[快照](../../history/20261105_workspace_handoff/documents_before/experiments/E101_final_c5_solver/README.md)。整理没有改模型、数值源码、冻结计划或任何运行证据。
