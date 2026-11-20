# E111_progressive_depth：固定dev500终末验证

唯一own-Loss best在Q评价前冻结。历史MIB及固定规格路径保留；测试未读，未按Q选权重。

```json
{
  "status": "complete",
  "count": 500,
  "checkpoint_sha256": "33cd9fd676b140aa9553864cf46bcd60f8c790ce04036fb6db0ce3e181cf50b4",
  "best_epoch": 35,
  "weighted_Q": 1.0469882098319838,
  "mean_Q": 1.057003687658303,
  "mixed_Q": 1.0519959487451436,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.2055585994005669,
  "p95_complete_seconds": 0.49663710500608427,
  "max_complete_seconds": 1.5251925000047777,
  "over250ms": 157,
  "max_Q": 2.700974153126065,
  "p95_Q": 1.1908166981167245,
  "training_selection_loss": 0.3694050078690052,
  "test_read": false,
  "selection_frozen_before_Q": true,
  "elapsed_seconds": 272.75895339998533
}
```

[同100例reference/L/最终图](gallery/index.html)
