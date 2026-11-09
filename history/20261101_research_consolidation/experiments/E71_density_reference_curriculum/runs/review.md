# E71：温和reference＋多尺度密度＋紧凑约束

一个seed0组合候选，保持96×12和16k。位置0.375降至0.25，紧凑0.5，新增密度0.5；reference仅监督位置/形状。唯一best只在第16轮起按稳定最终Loss选出；旧G和Q仅诊断，不跨阶段解释当期总Loss。复用历史E70及其他既有证据，不作单项因果归因。

| 组 | 分层 | 数量 | dev2000共同旧G | 交面积均值 |
|---|---|---:|---:|---:|
| E64_baseline | all | 2000 | 0.365703 | 7.146% |
| E64_baseline | low | 324 | 0.614409 | 9.715% |
| E64_baseline | middle | 868 | 0.425449 | 7.717% |
| E64_baseline | high | 808 | 0.201792 | 5.501% |
| E64_baseline | low_sparse | 157 | 0.788401 | 11.322% |
| E65_geometry | all | 2000 | 0.349914 | 6.241% |
| E65_geometry | low | 324 | 0.599549 | 8.707% |
| E65_geometry | middle | 868 | 0.408795 | 6.908% |
| E65_geometry | high | 808 | 0.186560 | 4.535% |
| E65_geometry | low_sparse | 157 | 0.775873 | 10.303% |
| E68_full_ports | all | 2000 | 0.390001 | 8.426% |
| E68_full_ports | low | 324 | 0.629408 | 10.479% |
| E68_full_ports | middle | 868 | 0.450831 | 9.056% |
| E68_full_ports | high | 808 | 0.228654 | 6.927% |
| E68_full_ports | low_sparse | 157 | 0.804916 | 12.024% |
| E70_taskloss | all | 2000 | 0.500023 | 3.820% |
| E70_taskloss | low | 324 | 0.742942 | 4.058% |
| E70_taskloss | middle | 868 | 0.584768 | 3.980% |
| E70_taskloss | high | 808 | 0.311577 | 3.553% |
| E70_taskloss | low_sparse | 157 | 0.954937 | 4.582% |
| full_ports | all | 2000 | 0.459483 | 4.972% |
| full_ports | low | 324 | 0.721260 | 5.671% |
| full_ports | middle | 868 | 0.546155 | 5.298% |
| full_ports | high | 808 | 0.261406 | 4.340% |
| full_ports | low_sparse | 157 | 0.932838 | 6.283% |

精确外框和空白比较（开发500，同例E70）：

| 模型 | B/A | 空白/B | 重复覆盖/A | 相对E70外框变化 | 外框增大例数 |
|---|---:|---:|---:|---:|---:|
| full_ports | 1.033486 | 8.086% | 5.014% | -3.195% | 1/500 |

[精确并集同例证据](geometry_vs_E70.json)。这张表与dev2000交面积使用不同面板，不混算。

同100例相对已接受E42的外框比较：

- full_ports：同100例外框平均变化-0.647%，22例增大，当前空白/B为8.133%。

[同100例精确几何证据](geometry_vs_E42_gallery100.json)。

组合方案相对历史参照的同例差及500例无端口模块误差、近邻相对误差/次序翻转见[review.json](review.json)。只作单seed自适应开发证据，不以flash Q重选checkpoint，不自动升级；草稿观感待用户评价。

![训练曲线](learning_curves.png)

- [full_ports同100例reference/current/E42无评分图](20261028_full_ports_density_curriculum_v1_seed0/draft_comparison100/index.html)

本候选完成即结束；不自动恢复旧长期研究或启动其他模型，测试保持封存。
