# E114 staged：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "staged",
  "count": 500,
  "checkpoint_sha256": "bd6906668cc2584133bb863590825874a951e7c24d8085280b7fd821c13f6937",
  "best_epoch": 48,
  "parameters": 612307,
  "weighted_Q": 1.0471514397614623,
  "mean_Q": 1.0281204204538439,
  "mixed_Q": 1.037635930107653,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.17604272019909695,
  "p95_complete_seconds": 0.4166630099876783,
  "max_complete_seconds": 1.4802802000194788,
  "over250ms": 119,
  "max_Q": 3.6781861758585275,
  "p95_Q": 1.0973704066612844,
  "raw_repeated_coverage": 0.05199379508982779,
  "raw_bbox_over_area": 1.0262404956339437,
  "common_final_dev_loss": 0.3423073971569538,
  "own_selection_loss": 0.36958171063661577,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 6561.172119900002,
  "training_epochs": 49,
  "elapsed_seconds": 183.09395399998175,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
