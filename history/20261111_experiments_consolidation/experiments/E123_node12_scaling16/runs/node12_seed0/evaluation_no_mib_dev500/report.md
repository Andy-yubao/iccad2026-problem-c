# E123_node12_scaling16 node12：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "node12",
  "count": 500,
  "checkpoint_sha256": "79357ba02ec3e1351895b46e97994060a543b9f2ce809babdc6d7592fe4ae417",
  "best_epoch": 67,
  "parameters": 932227,
  "weighted_Q": 1.0605463380903288,
  "mean_Q": 1.0390535281725453,
  "mixed_Q": 1.0497999331314372,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.19351983279862905,
  "p95_complete_seconds": 0.4761725999909683,
  "max_complete_seconds": 1.58351650001714,
  "over250ms": 125,
  "max_Q": 3.6245093373790183,
  "p95_Q": 1.1250458041117615,
  "raw_repeated_coverage": 0.03484522470460401,
  "raw_bbox_over_area": 1.0241728903475247,
  "common_final_dev_loss": 0.349705459728837,
  "own_selection_loss": 0.3777506868243217,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 7616.82785219996,
  "training_epochs": 67,
  "elapsed_seconds": 213.99702160002198,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
