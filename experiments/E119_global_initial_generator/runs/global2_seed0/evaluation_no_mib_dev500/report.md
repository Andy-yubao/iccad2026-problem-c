# E119_global_initial_generator global2：固定dev500验证

唯一own-Loss best在Q评价前冻结；原始面积/无MIB，全模块真实草稿。

```json
{
  "status": "complete",
  "arm": "global2",
  "count": 500,
  "checkpoint_sha256": "5be725ec4394f696fb07e7d6df08eadd81ab0c6d1de738a410250c1659ed6799",
  "best_epoch": 55,
  "parameters": 712275,
  "weighted_Q": 1.0212682755773015,
  "mean_Q": 1.021867346847345,
  "mixed_Q": 1.0215678112123232,
  "hard_legal": 500,
  "raw_hard_legal": 0,
  "fallbacks": 0,
  "mean_complete_seconds": 0.19338800760079175,
  "p95_complete_seconds": 0.4977803349756868,
  "max_complete_seconds": 1.2757660999777727,
  "over250ms": 137,
  "max_Q": 1.7095417333208534,
  "p95_Q": 1.090189526831662,
  "raw_repeated_coverage": 0.04794949825008766,
  "raw_bbox_over_area": 1.0243355565467578,
  "common_final_dev_loss": 0.3331280593574047,
  "own_selection_loss": 0.356844794601202,
  "scoring_policy": "local research Q without MIB, original areas/specs; not official contest score",
  "training_seconds": 7115.031832600012,
  "training_epochs": 57,
  "elapsed_seconds": 221.4611348999897,
  "test_read": false,
  "selection_frozen_before_Q": true
}
```
