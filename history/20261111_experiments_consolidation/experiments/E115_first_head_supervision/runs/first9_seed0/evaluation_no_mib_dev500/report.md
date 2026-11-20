# E115 first9：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "first9",
  "count": 500,
  "checkpoint_sha256": "dc4768bbd5ee72b43e37f2280a8396d45c44684702cebd18f4578527f5cfad42",
  "best_epoch": 50,
  "parameters": 609682,
  "weighted_Q": 1.0447081733350936,
  "mean_Q": 1.0357106586619784,
  "mixed_Q": 1.040209415998536,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.18731888859753962,
  "p95_complete_seconds": 0.44226104503031805,
  "max_complete_seconds": 1.4326828000484966,
  "over250ms": 132,
  "max_Q": 2.833414957002861,
  "p95_Q": 1.1274403608720627,
  "raw_repeated_coverage": 0.0664065049515948,
  "raw_bbox_over_area": 1.0232462235838982,
  "common_final_dev_loss": 0.3565098087489605,
  "own_selection_loss": 0.4295977640151978,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 5601.021995600022,
  "training_epochs": 50,
  "elapsed_seconds": 184.19712450000225,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
