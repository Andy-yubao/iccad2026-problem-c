# E122_global2_scaling16 global2：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "global2",
  "count": 500,
  "checkpoint_sha256": "785447370a26426cebfe216ed0310dd66e67020aae91c7b0c5d010cc9211532c",
  "best_epoch": 57,
  "parameters": 914435,
  "weighted_Q": 1.026367486087441,
  "mean_Q": 1.0260423713213076,
  "mixed_Q": 1.0262049287043742,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.18652841419971083,
  "p95_complete_seconds": 0.4385971000010612,
  "max_complete_seconds": 1.5963977999635972,
  "over250ms": 132,
  "max_Q": 1.6839631648976088,
  "p95_Q": 1.1009985495484302,
  "raw_repeated_coverage": 0.04828312767339517,
  "raw_bbox_over_area": 1.0182894982107908,
  "common_final_dev_loss": 0.3307334144115448,
  "own_selection_loss": 0.35510994869470597,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 9326.585999400006,
  "training_epochs": 57,
  "elapsed_seconds": 208.52622830000473,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
