# E125_global2_ref04 global2：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "global2",
  "count": 500,
  "checkpoint_sha256": "e3884491f7f5e262373c99dc69c5e964294a2b4fb2e4479279f584867a8c92c3",
  "best_epoch": 53,
  "parameters": 712275,
  "weighted_Q": 1.0337279002063047,
  "mean_Q": 1.0375048348652207,
  "mixed_Q": 1.0356163675357628,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.1923278941998724,
  "p95_complete_seconds": 0.4898704500024905,
  "max_complete_seconds": 1.3197987000457942,
  "over250ms": 135,
  "max_Q": 2.4709656985786217,
  "p95_Q": 1.1313959174890587,
  "raw_repeated_coverage": 0.04702707038715778,
  "raw_bbox_over_area": 1.020444107782942,
  "common_final_dev_loss": 0.3737817824333906,
  "own_selection_loss": 0.21556613746285438,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 7106.512084499991,
  "training_epochs": 56,
  "elapsed_seconds": 216.72816379996948,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
