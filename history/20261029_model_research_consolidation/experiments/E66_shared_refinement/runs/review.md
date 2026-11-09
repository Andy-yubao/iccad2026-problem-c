# E66：增量输出与共享迭代修正

全部固定seed0；用户明确不做多种子确认。原八项G及数据/训练规则冻结。

| 组 | best轮 | train500 G | dev500 G | dev2000 G | raw交面积均值 | flash加权Q |
|---|---:|---:|---:|---:|---:|---:|
| E64 baseline | 40 | 0.304448 | 0.368244 | 0.365703 | 7.146% | 1.030026 |
| unshared_residual | 42 | 0.337515 | 0.379448 | 0.376591 | 7.248% | 1.039248 |
| shared_residual | 39 | 0.342698 | 0.381916 | 0.378314 | 7.223% | 1.038051 |

两组均12次块应用，分别12个独立块与3块循环4次；不叠加E65消息。E64到unshared同时改变增量表示、步幅与head归一化，不能区分各自贡献。E66组间固定有效深度但参数量不同，需同时观察train/dev，不只看差距。common_steps及完整分项见[review.json](review.json)。本报告不以Q重选模型，不自动升级E42。数值候选门槛仅供决定后续机制验证，草稿视觉评价待用户。

unshared_residual：固定train/dev G相对E64为+10.86%/+3.04%，raw交面积+1.43%；预设数值候选门槛未满足。

shared_residual：固定train/dev G相对E64为+12.56%/+3.71%，raw交面积+1.08%；预设数值候选门槛未满足。

![共同更新数曲线](learning_curves.png)

- [unshared_residual同100例reference/current/E42无评分图](20261028_unshared_residual_seed0/draft_comparison100/index.html)
- [shared_residual同100例reference/current/E42无评分图](20261028_shared_residual_seed0/draft_comparison100/index.html)

单种子、自适应开发证据，不是正式比赛成绩或独立泛化结论。测试封存，原权重保留。长期任务按[long_task.md](../../E65_geometry_messages/long_task.md)继续，由每小时唤醒检查并准备下一阶段。


## 2026-10-28阶段复核与停止扩展

两组正常早停、flash均500/500硬合法，原保护权重未改。固定train G比E64高10.86%/12.56%，dev G高3.04%/3.71%；train—dev差距从E64的0.06380缩至0.04193/0.03922，是伴随拟合变差的缩小，不能称为泛化收益。当前有界增量表示/新head归一化/迭代的组合未获益，组间共享也未改善dev；这些证据不否定所有共享迭代架构。

抽看unshared case10/16/20及shared case10/16共5张既有图，宏观参考组织仍在，但局部冲突仍明显，未看到一致优势；不能代替用户图评。停止当前E66配置扩展，不追加调参、重训或多种子；保留失败方向及唯一checkpoint，下一步独立E67多尺度，不叠加E65/E66。测试封存。
