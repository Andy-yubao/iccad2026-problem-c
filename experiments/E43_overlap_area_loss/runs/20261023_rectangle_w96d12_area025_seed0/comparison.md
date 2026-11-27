# E43 重叠面积训练与冻结E42联合AB对照

保留E42模型及AB，仅新增0.25×重叠面积训练梯度，两轮渐入；不做数据增强。原geometry＋occupancy选模代理与MIB处理不变，新total不能与旧total比较。单种子开发，不按终末Q反选checkpoint。

| 模型 | 状态 | 轮数 | best轮 | monitor代理 |
|---|---|---:|---:|---:|
| E42 AB | early_stop | 54 | 51 | 0.454408692 |
| E43 AB＋area | early_stop | 46 | 44 | 0.451486322 |

| 模型 | full-dev代理 | 硬合法 | 加权Q | 算术Q | P95 | 最差5% | 最大Q | 秒/例 | 救援 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E42 AB | 0.45381430387496946 | 500/500 | 1.025860 | 1.046002 | 1.185535 | 1.291023 | 1.927279 | 0.251334 | 4 |
| E43 AB＋area | 0.4487406511306763 | 500/500 | 1.029275 | 1.040796 | 1.155637 | 1.251210 | 1.815712 | 0.260098 | 5 |

加权Q配对差/区间：{'weighted_Q_delta': 0.003415582373528834, 'case_bootstrap_95': [-0.0005365219461708124, 0.00777679482578422], 'worker_cluster_bootstrap_95': [-0.0008602174871642512, 0.007475622247015074], 'Q_better_equal_worse': [229, 57, 214], 'limitation': 'One seed and one timed repair run; intervals do not cover training seeds or inter-run timing.'}
同时核对full_dev的bbox、compactness、void、boundary、contact、MIB，防止靠摊大布局换重叠下降。

固定分层及总体配对结果见comparison.json；预放置/重叠/旧难例分层不改变全体评价分母。
旧模型条件化难例存在均值回归，4/12/22例子组只作诊断；病例配对不覆盖运行间波动和种子不确定性。
