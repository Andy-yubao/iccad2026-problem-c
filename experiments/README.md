# 实验入口

| 目录 | 角色 |
|---|---|
| `E119_global_initial_generator/` | 主要研究基线global2 best55；含独立public_priority_v1变体和已授权test500历史评价 |
| `E120_node12_collision/` | Node12结构对照，训练/原dev500已完成 |
| `E131_relation_input/` | 旧64×4关系输入，用户停止，不自动恢复 |
| `E132_relation_input_96x6/` | full/pure训练已完成，无后处理/Q授权 |
| `E133_flow_matching/` | FM训练已完成；原自动评价失败现场保留，独立成功评价在evaluation_quality_20261113 |
| `E134_supervision_interval/` | E119间隔2/3两组已完成训练/原dev500；E120间隔3门禁停止，用户放弃后续E120，间隔2未启动；保留失败现场 |
| `E135_initial_supervision/` | E119 S0监督 A/B/C及自动原dev500后处理全部完成，均500/500最终硬合法；结果在runs/results_review.md，S0–S12展示在各run/visualization_dev15_loss |
| `E136_frontend_F_linear/` | D：训练和原dev500已完成，best46/混合Q1.021426；Reference与五组草稿/原后处理的逐题统计在runs/statistics_dev500_v1 |

具体身份看 `plan.json`、`authorization.json`、冻结权重记录；实际状态看 `queue_state.json` 和run内结果。文档和旧true开关不构成新的启动授权。长期结果留所属实验，公用数据/评分/展示不在实验间复制调用。

退役实验实际位置见 [完整目录](../docs/experiment_catalog.md)。E101交付及E98、E102–E118、E121–E130位于 `history/20261111_experiments_consolidation/experiments/`。本次整理未恢复旧实验、训练或评价。
