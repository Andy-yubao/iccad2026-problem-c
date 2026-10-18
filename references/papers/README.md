# ML 布局规划论文精选

2026-10-16（北京时间）补充。**先读 [中文综述](literature_review.zh-CN.md)**：包含方法路线图、逐篇阅读卡、实验结论的适用范围，以及面向 Problem C 的候选假设。关键结论附有本地 PDF 页码。

本目录共 11 篇：新增下载 10 篇，复用已有 FloorSet 原件。阅读重点是主要方法、核心实验与影响结论的附录；原文性能均未在本地复现。

| 论文与本地 PDF | 版本/来源 | 页数 | 阅读价值 |
|---|---|---:|---|
| [FloorSet — a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs](FloorSet_2405.05480.pdf) | 已有 arXiv v4，2024 | 10 | 数据、标签、约束背景 |
| [Trinity: One Differentiable Physics for Training, Refining and Scoring Generative Floorplanners](Trinity_2610.04957v1.pdf) | arXiv v1，2026-10-04，预印本 | 85 | 同题约束建模、分阶段消融；必须看数据与候选选择限定 |
| [Chip Placement with Diffusion Models](ChipDiffusion_ICML2025.pdf) | ICML 2025 / PMLR | 16 | 图条件扩散、局部与全局交互、引导 |
| [FlowPlace: Flow Matching for Chip Placement](FlowPlace_2604.23658v2.pdf) | arXiv v2，标注 DAC 2026 | 7 | 少步生成与采样中的几何投影 |
| [DiffPlace: A Conditional Diffusion Framework for Simultaneous VLSI Placement Beyond Sequential Paradigms](DiffPlace_2510.15897v2.pdf) | arXiv v2，2026-04-07 | 16 | 相对几何消息、质量条件与局部引导 |
| [MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning](MaskPlace_NeurIPS2022.pdf) | NeurIPS 2022 | 12 | 可行位置掩码、线长增量图 |
| [ChiPFormer: Transferable Chip Placement via Offline Decision Transformer](ChiPFormer_ICML2023.pdf) | ICML 2023 / PMLR | 19 | 离线轨迹学习、跨电路迁移 |
| [Learn to Floorplan through Acquisition of Effective Local Search Heuristics](LearnToFloorplan_ICCD2020.pdf) | ICCD 2020，作者托管原文 | 8 | sequence pair、候选动作价值学习 |
| [RulePlanner: All-in-One Reinforcement Learner for Unifying Design Rules in 3D Floorplanning](RulePlanner_2601.22476v2.pdf) | arXiv v2，标注 ICML 2026 | 28 | 多规则矩阵、位置与形状联合动作 |
| [DC3: A Learning Method for Optimization with Hard Constraints](DC3_2104.12225.pdf) | arXiv v1，正文标注 ICLR 2021 | 17 | 等式补全、不等式修正与保证边界 |
| [PARSAC: Fast, Human-quality Floorplanning for Modern SoCs with Complex Design Constraints](PARSAC_2405.05495v3.pdf) | arXiv v3，2024-08-01 | 9 | 传统约束处理、锚定、专用修复动作 |

共 227 页，含参考文献与附录。这里的“标注”来自 arXiv 元数据或 PDF，本轮没有逐篇额外核验会议录收录状态；PMLR/NeurIPS 下载的是正式会议录版本。DiffPlace 的期刊模板有卷期占位符，不能据此认定正式发表。

## 来源与完整性

[manifest.json](manifest.json) 保存题名、原始页面、PDF 请求与实际跳转 URL、下载时间、版本、字节数、SHA-256、可读页数及阅读核对范围。新论文来自 arXiv、PMLR 官方 PDF 仓库、NeurIPS 或作者主页；没有将演示稿、搜索摘要和付费全文占位页当成已读论文。

FloorSet 的 SHA-256 仍为 `6552d7a7befd804458056548c56d9dd0d626e053db13202191b128441811ac99`，与初始 [资源清单](../resources_manifest.json) 相同；旧清单保持原样。原件保留原署名与许可，不代表本地综述拥有原文版权。

资料获取入口是 [fetch_literature.py](../../scripts/fetch_literature.py)，在工作区根目录使用现有独立环境运行：

```powershell
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
python -B scripts/fetch_literature.py
```

已有文件只核验，不覆盖；缺失的新论文按固定来源获取。阅读抽取文本和页面渲染仅作临时核对，不作为第二套论文原件保存。模型、权重、额外数据和实验输出不在本目录。

返回 [资料总索引](../README.md)。
