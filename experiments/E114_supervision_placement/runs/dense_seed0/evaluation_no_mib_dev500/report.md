# E114 dense：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "dense",
  "count": 500,
  "checkpoint_sha256": "5a709e22823d3e39c44791170ec2ea40248259cdbb3d0e7833a4d3a79c1f4b7f",
  "best_epoch": 52,
  "parameters": 612307,
  "weighted_Q": 1.04599190332473,
  "mean_Q": 1.0286820651778035,
  "mixed_Q": 1.0373369842512667,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.17636383800016484,
  "p95_complete_seconds": 0.40672352000110523,
  "max_complete_seconds": 1.582182299986016,
  "over250ms": 119,
  "max_Q": 3.2281307436476765,
  "p95_Q": 1.1064749383296923,
  "raw_repeated_coverage": 0.050130356158396,
  "raw_bbox_over_area": 1.02266614198188,
  "common_final_dev_loss": 0.3415501182079315,
  "own_selection_loss": 0.37901437932252885,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 13198.2202597,
  "training_epochs": 53,
  "elapsed_seconds": 188.2731347999943,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
