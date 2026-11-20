# E110_deep_supervision_heads：固定dev500终末验证

唯一own-Loss best在Q评价前冻结。历史MIB及固定规格路径保留；测试未读，未按Q选权重。

```json
{
  "status": "complete",
  "count": 500,
  "checkpoint_sha256": "8f8641fc872bdca2be618719de193eb383d97ed80ef2e122069c8f7365a50c40",
  "best_epoch": 56,
  "weighted_Q": 1.0353811400276252,
  "mean_Q": 1.0410974722434698,
  "mixed_Q": 1.0382393061355475,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.20246106359903934,
  "p95_complete_seconds": 0.48072420500539,
  "max_complete_seconds": 1.9325600999873132,
  "over250ms": 153,
  "max_Q": 2.109275839577046,
  "p95_Q": 1.1436912534910002,
  "training_selection_loss": 0.41132047897577284,
  "test_read": false,
  "selection_frozen_before_Q": true,
  "elapsed_seconds": 284.89492409999366
}
```

[同100例reference/L/最终图](gallery/index.html)
