# 2026-11-11 实验区收敛归档

用户要求按工作区整理规范把 `experiments/` 整理为仅保留 **E119_global_initial_generator**（主要研究基线，`protocol/research_baseline_v3.json`）与 **E120_node12_collision**（Node12结构对照）。确认所有队列 `queue_state.json` 均为 complete、记录的控制器PID均不存活、移动前30分钟 experiments 下零文件写入、全树 junction/symlink 数量为 0 后，29 个退役实验目录按登记清单逐目录同盘 rename 迁入本归档；E119/E120、`README.md`、`__init__.py` 留在 experiments，`experiments/__pycache__`（唯一 .pyc）按可重建缓存删除。

## 核对方式

- 移动前后各生成一次逐文件清单（相对路径、字节大小、UTC mtime；≤8MiB 文件另算 SHA256），文件集合、大小、SHA256 全部一致，mtime 解析后最大差 1µs（时间戳精度舍入）。证据：[verification/](verification/)，结论见 [manifest.json](manifest.json) 与 `verification/verify_result.txt`。
- 共 **145,392 个文件、256,551,060,683 字节（约256.6GB）**，其中 140,906 个小文件 SHA256 两侧逐位相等。
- 移动为同卷 rename，不改写内容；旧 JSON、源码、日志、授权与命令保持发生时原样，未做任何路径批改。

## 迁移映射

| 实验 | 文件数 | 大小 | 说明 |
|---|---|---|---|
| [E101_final_c5_solver](experiments/E101_final_c5_solver/README.md) | 67,348 | 250.3GB | 原C5交付与重训证据；体积主要来自 replay/chain64k 隔离运行根 |
| [E102_reference_image_generation](experiments/E102_reference_image_generation/README.md) | 34,247 | 1.6GB | 图片兴趣实验，已停止 |
| [E103_offline_loss_alignment](experiments/E103_offline_loss_alignment/README.md) | 8,615 | 0.9GB | 离线Loss对齐研究 |
| [E98_loss_weight_learning](experiments/E98_loss_weight_learning/README.md) | 10,721 | 0.9GB | 已放弃的调权搜索；2026-11-04曾迁回，本次再归档 |
| [E106_postprocessor_aware_loss](experiments/E106_postprocessor_aware_loss/README.md) | 10,110 | 1.2GB | 代理Loss方向，未训练布局器 |
| [E104_scaling32k](experiments/E104_scaling32k/README.md) | 788 | 270MB | 扩容泛化增强版 |
| [E105_gated_prenorm_placer](experiments/E105_gated_prenorm_placer/README.md) | 738 | 139MB | 方案一候选 |
| [E107_geometry_posttrain](experiments/E107_geometry_posttrain/README.md) | 147 | 120MB | 门禁停止，未终末评价 |
| [E108_random_reference40k](experiments/E108_random_reference40k/README.md) | 23 | 0.2MB | 随机40k数据准备入口（数据本体在 datasets/） |
| [E109_deep_supervision_shared](experiments/E109_deep_supervision_shared/README.md) | 753 | 87MB | A：共享头逐层监督（旧重要基线） |
| [E110_deep_supervision_heads](experiments/E110_deep_supervision_heads/README.md) | 743 | 89MB | B：独立头逐层监督 |
| [E111_progressive_depth](experiments/E111_progressive_depth/README.md) | 780 | 146MB | 4→8→12分阶段 |
| [E112_deep_geometry_budget](experiments/E112_deep_geometry_budget/README.md) | 136 | 41MB | 几何MLP加深（暂停） |
| [E113_deep_head_budget](experiments/E113_deep_head_budget/README.md) | 111 | 0.5MB | 共享头加深（未启动） |
| [E114_supervision_placement](experiments/E114_supervision_placement/README.md) | 2,252 | 111MB | 监督位置三组＋大预算评价 |
| [E115_first_head_supervision](experiments/E115_first_head_supervision/README.md) | 616 | 35MB | 九层1+9大首头 |
| [E116_first_draft_comparison](experiments/E116_first_draft_comparison/README.md) | 73 | 20MB | 首稿结构对照 |
| [E117_node26](experiments/E117_node26/README.md) | 631 | 42MB | Node26节点几何投影 |
| [E118_first12_supervision](experiments/E118_first12_supervision/README.md) | 595 | 33MB | 1+12监督（E119前置） |
| [E121_residual_update](experiments/E121_residual_update/README.md) | 599 | 35MB | 状态S残差递推 |
| [E122_global2_scaling16](experiments/E122_global2_scaling16/README.md) | 597 | 40MB | E119扩深16几何块 |
| [E123_node12_scaling16](experiments/E123_node12_scaling16/README.md) | 601 | 42MB | E120扩深16几何块 |
| [E124_residual_scaling16](experiments/E124_residual_scaling16/README.md) | 599 | 54MB | E121扩深16几何块（含恢复证据） |
| [E125_global2_ref04](experiments/E125_global2_ref04/README.md) | 597 | 36MB | 监督首轮：reference .4/.4 |
| [E126_global2_refsplit](experiments/E126_global2_refsplit/README.md) | 593 | 36MB | 监督首轮：1/.25 |
| [E127_node12_ref04](experiments/E127_node12_ref04/README.md) | 594 | 36MB | E120结构 .4/.4 |
| [E128_node12_refsplit](experiments/E128_node12_refsplit/README.md) | 594 | 38MB | E120结构 1/.25 |
| [E129_global2_s0ref](experiments/E129_global2_s0ref/README.md) | 597 | 37MB | 三处监督：S0纯reference（E119结构） |
| [E130_node12_s0ref](experiments/E130_node12_s0ref/README.md) | 594 | 36MB | 三处监督：S0纯reference（E120结构） |

## 阅读与使用注意

- **MAX_PATH**：E101 的 `runs/20261103_replay` 与 `runs/20261104_chain64k_seed0` 内 s42/s74/sf 隔离运行根整棵复制了历史缓存结构，归档后前缀变长，部分深层路径超过 Win32 260 字符限制，PowerShell 5.1/资源管理器可能报"找不到路径"。文件完整存在（移动前清单无任何访问错误），读取该子树请用支持长路径的工具（Python 3.6+ 或 `\\?\` 前缀）。
- 历史 README/JSON/命令保留发生时路径；按 manifest 映射解释，不直接执行旧启动入口，不为"修链接"批改冻结证据。
- docs/output/visualization 历史条目中的 `experiments/<archived>` 相对链接按本 manifest 解释；`.gitignore` 中指向旧路径的规则成为 no-op，保留不动。
- E42/E58 等更早归档缓存映射（见 `docs/workspace_structure.md`）不受影响；`datasets/static_tensor_cache_v1` 共享缓存与 protocol/ 索引原位。
- 恢复任何归档实验须先获用户明确授权并登记新入口/读取适配，不能绕过身份检查。
- 本次整理未做 git 提交；tracked 源文件的删除显示在工作区改动中，由用户决定提交时机。

一次性核查/迁移脚本原件在 [scripts/](scripts/)。
