# E70：新任务Loss＋完整端口

新增一个seed0候选，沿用E68完整端口结构，从头训练新固定Loss；reference仅监督位置/形状。唯一best按新Loss选出，旧G仅为共同诊断指标，不与历史总Loss混比，不根据Q或旧G重选。对照复用已完成E64/E65/E68，无多种子。

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
| full_ports | all | 2000 | 0.500023 | 3.820% |
| full_ports | low | 324 | 0.742942 | 4.058% |
| full_ports | middle | 868 | 0.584768 | 3.980% |
| full_ports | high | 808 | 0.311577 | 3.553% |
| full_ports | low_sparse | 157 | 0.954937 | 4.582% |

组合方案相对历史参照的同例差及500例无端口模块误差、近邻相对误差/次序翻转见[review.json](review.json)。只作单seed自适应开发证据，不以flash Q重选checkpoint，不自动升级；草稿观感待用户评价。

![训练曲线](learning_curves.png)

- [full_ports同100例reference/current/E42无评分图](20261028_full_ports_taskloss_v1_seed0/draft_comparison100/index.html)

本候选完成即结束；不自动恢复旧长期研究或启动其他模型，测试保持封存。
