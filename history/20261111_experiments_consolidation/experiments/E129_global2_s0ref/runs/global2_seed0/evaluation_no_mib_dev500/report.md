# E129_global2_s0ref global2：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "global2",
  "count": 500,
  "checkpoint_sha256": "576fff3c10cbf824f42838aa7f11a21394fc72df02540cd3e34ee90b96c8726c",
  "best_epoch": 67,
  "parameters": 712275,
  "weighted_Q": 1.048339146852552,
  "mean_Q": 1.030780923571658,
  "mixed_Q": 1.039560035212105,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.18015542919980362,
  "p95_complete_seconds": 0.4319967550865839,
  "max_complete_seconds": 1.4466595000121742,
  "over250ms": 126,
  "max_Q": 3.727517763375893,
  "p95_Q": 1.1051709180756477,
  "raw_repeated_coverage": 0.04331879438485207,
  "raw_bbox_over_area": 1.0234580215289646,
  "common_final_dev_loss": 0.33513064762949946,
  "own_selection_loss": 0.308667323410511,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 8704.825784799992,
  "training_epochs": 70,
  "elapsed_seconds": 198.63768719998188,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
