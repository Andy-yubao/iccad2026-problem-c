# E98泛化增强基线：冻结best71固定深度dev500评价

- 加权Q：1.043560018
- 平均Q：1.050714113
- 混合Q：1.047137066
- 硬合法：500/500；回退：0
- 逐例完整耗时：均值203.488ms / P95 488.322ms / 最大1437.873ms
- 超250ms：157/500
- train500 Loss=0.371603540；dev500 Loss=0.375844625；相对差距=1.14%

模型在Q评价前按固定验证Loss冻结，未按Q重选。dev已用于训练监测，本结果不是未见测试或正式比赛成绩。
历史E98使用时间预算后处理，本次使用计算深度版本，不能直接据两者Q差异归因于泛化增强。
测试未读取，调权未启动。初始化单列；评分、磁盘和绘图不冒充逐例求解耗时。

[同100例reference/L/最终布局](gallery/index.html)

```json
{
  "count": 500,
  "weighted_Q": 1.0435600181008373,
  "mean_Q": 1.0507141132280862,
  "mixed_Q": 1.0471370656644616,
  "hard_legal": 500,
  "mean_complete_seconds": 0.20348829099960858,
  "p95_complete_seconds": 0.48832214000722163,
  "max_complete_seconds": 1.4378727999865077,
  "over250ms": 157,
  "p95_Q": 1.1589587060852162,
  "max_Q": 3.894841455281734,
  "fallbacks": 0,
  "status": "complete",
  "checkpoint_sha256": "3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea",
  "model_hash": "67e4ca2a8c01c8f06656b1d9c0b9967cf7fae3ed6874fda2452b546d65539010",
  "best_epoch": 71,
  "raw_hard_legal": 0,
  "mean_draft_module_pass10": 0.42486358900236154,
  "train_monitor_loss": 0.37160353979468347,
  "dev_monitor_loss": 0.37584462502598764,
  "relative_loss_gap": 0.011412930118070097,
  "elapsed_seconds": 215.21175990000484,
  "test_read": false,
  "quality_scope": "official neutral-time Q; mixedQ=(weightedQ+meanQ)/2; complete per-case timing includes public input construction, transfer, model, repair and independent hard checks; initialization/scoring/disk/plots separate",
  "comparison_limit": "Historical E98 Q used wall-budget repair; not directly comparable to this depth-budget result",
  "selection_frozen_before_Q": true
}
```

补充说明：本地公开MIB平均面积派生任务，非正式比赛成绩。原固定基线train/dev Loss=0.341676799/0.354983613，相对差距3.89%；新基线差距1.14%，但验证Loss绝对值较旧基线高5.88%。差距缩小本身不能证明最终布局质量提升。

质量尾部：P95 Q=1.158958706，最大Q=3.894841455；草稿硬合法0/500，最终后处理硬合法500/500。
