# E68 v2：完整端口组合方案

新增训练只有一个seed0候选，固定G，开发G选唯一best。组合包括移除旧端口摘要、完整逐边输入和保留权重绝对强度的聚合。复用E64/E65历史结果，仅评估组合总体是否改善，不归因到某一组件；取得明显进步后再考虑回溯消融。

| 组 | 分层 | 数量 | dev2000分层G | 交面积均值 |
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
| full_ports | all | 2000 | 0.390001 | 8.426% |
| full_ports | low | 324 | 0.629408 | 10.479% |
| full_ports | middle | 868 | 0.450831 | 9.056% |
| full_ports | high | 808 | 0.228654 | 6.927% |
| full_ports | low_sparse | 157 | 0.804916 | 12.024% |

组合方案相对历史参照的同例差及500例无端口模块误差、近邻相对误差/次序翻转见[review.json](review.json)。只作单seed自适应开发证据，不以flash Q重选checkpoint，不自动升级；草稿观感待用户评价。

![训练曲线](learning_curves.png)

- [full_ports同100例reference/current/E42无评分图](20261028_full_ports_nosummary_v2_seed0/draft_comparison100/index.html)

本候选完成即结束；不自动恢复旧长期研究或启动其他模型，测试保持封存。

[训练结果复核与结论](inspection.md)
