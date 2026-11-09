# E114 staged：3倍后处理预算固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "staged",
  "count": 500,
  "checkpoint_sha256": "bd6906668cc2584133bb863590825874a951e7c24d8085280b7fd821c13f6937",
  "best_epoch": 48,
  "parameters": 612307,
  "weighted_Q": 1.0218204198873524,
  "mean_Q": 1.0214565256232786,
  "mixed_Q": 1.0216384727553156,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.34184399680036587,
  "p95_complete_seconds": 1.027282850036863,
  "max_complete_seconds": 3.654026700009126,
  "over250ms": 241,
  "over500ms": 82,
  "over1s": 28,
  "max_Q": 1.5055735725408455,
  "p95_Q": 1.0918968048914144,
  "raw_repeated_coverage": 0.05199379508982779,
  "raw_bbox_over_area": 1.0262404956339437,
  "common_final_dev_loss": 0.3423073971569538,
  "own_selection_loss": 0.36958171063661577,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 6561.172119900002,
  "training_epochs": 49,
  "elapsed_seconds": 281.24359040003037,
  "test_read": false,
  "selection_frozen_before_Q": true,
  "target_mean_complete_seconds": 0.5,
  "work_budget_multiplier": 3,
  "wall_clock_cutoff": false,
  "mean_repair_seconds": 0.33123084979876877,
  "exact_draft_matches": 500
}
```
