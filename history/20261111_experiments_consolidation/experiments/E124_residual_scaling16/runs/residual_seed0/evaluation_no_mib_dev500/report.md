# E124_residual_scaling16 residual：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "residual",
  "count": 500,
  "checkpoint_sha256": "99038efdbb27f0fb8e496f430f088bc6f6e0d955b43728d1aa9a6434645db001",
  "best_epoch": 46,
  "parameters": 917555,
  "weighted_Q": 1.0338837073802447,
  "mean_Q": 1.030586120453659,
  "mixed_Q": 1.0322349139169518,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.17317971999885048,
  "p95_complete_seconds": 0.40605399497435407,
  "max_complete_seconds": 1.4051372000249103,
  "over250ms": 119,
  "max_Q": 2.0047086105078584,
  "p95_Q": 1.1177609236804589,
  "raw_repeated_coverage": 0.047215877203552824,
  "raw_bbox_over_area": 1.020173811507629,
  "common_final_dev_loss": 0.3495257107615471,
  "own_selection_loss": 0.3790452139079571,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 8138.731596899976,
  "training_epochs": 49,
  "elapsed_seconds": 178.9451547000208,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
