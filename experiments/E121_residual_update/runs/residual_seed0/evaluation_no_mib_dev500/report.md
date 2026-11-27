# E121_residual_update residual：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "residual",
  "count": 500,
  "checkpoint_sha256": "fa9eae72c68347b209c3fa39dc5a9574854c91212ece64e331068a6851ff73dc",
  "best_epoch": 53,
  "parameters": 714615,
  "weighted_Q": 1.051716936960119,
  "mean_Q": 1.0300764599605285,
  "mixed_Q": 1.0408966984603238,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.18576292299840133,
  "p95_complete_seconds": 0.45704459997068553,
  "max_complete_seconds": 1.9840883000288159,
  "over250ms": 125,
  "max_Q": 3.5838540905828746,
  "p95_Q": 1.1071162409358586,
  "raw_repeated_coverage": 0.04614791459726608,
  "raw_bbox_over_area": 1.021276935464715,
  "common_final_dev_loss": 0.33362095268070696,
  "own_selection_loss": 0.3613037246763706,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 7036.812317700009,
  "training_epochs": 56,
  "elapsed_seconds": 215.08986820001155,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
