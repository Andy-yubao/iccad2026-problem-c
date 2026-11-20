# E98首轮训练集求解反馈调权

{
  "selected": {
    "status": "frozen_before_dev",
    "selected_arm": "plus",
    "checkpoint": "experiments/E98_loss_weight_learning/runs/20261102_weight_round1_seed0/branches/plus/checkpoints/latest.pt",
    "checkpoint_sha256": "94c5086a354a444e28016337baa37f42a777cbd9aafc1d60674fbfcee708839c",
    "coefficients": {
      "position": 1.1862203142327061,
      "shape": 0.29655507855817653,
      "penetration": 0.0681377312658052,
      "module_load_mean": 0.0681377312658052,
      "module_max_excess10": 0.10646520510282063,
      "wirelength": 0.1979109016045599,
      "compactness": 0.12666297702691837,
      "boundary_contribution": 0.7916436064182396,
      "grouping_contribution": 0.5066519081076735,
      "mib_contribution": 0.5066519081076735
    },
    "coefficient_changed": true,
    "accepted_candidates": [
      "plus"
    ],
    "feedback_source": "24 frozen train members only, two repeats, no dev reads",
    "selection_rule": "candidate must beat control by >0.0001 on BOTH feedback repeats",
    "validation_may_change_selection": false,
    "test_read": false
  },
  "train_feedback": {
    "base": [
      {
        "repeat": 0,
        "weighted_Q": 1.0417365034730506,
        "all_legal": true,
        "mean_seconds": 0.24251862500144247
      },
      {
        "repeat": 1,
        "weighted_Q": 1.0393578561209826,
        "all_legal": true,
        "mean_seconds": 0.22510604166745907
      }
    ],
    "control": [
      {
        "repeat": 0,
        "weighted_Q": 1.0322966812970373,
        "all_legal": true,
        "mean_seconds": 0.23849582916530684
      },
      {
        "repeat": 1,
        "weighted_Q": 1.0346661744266958,
        "all_legal": true,
        "mean_seconds": 0.22737623749950822
      }
    ],
    "plus": [
      {
        "repeat": 0,
        "weighted_Q": 1.0297617797468928,
        "all_legal": true,
        "mean_seconds": 0.2534332916669276
      },
      {
        "repeat": 1,
        "weighted_Q": 1.0326598912040517,
        "all_legal": true,
        "mean_seconds": 0.2649588291654557
      }
    ],
    "minus": [
      {
        "repeat": 0,
        "weighted_Q": 1.038180229667009,
        "all_legal": true,
        "mean_seconds": 0.2197634249987459
      },
      {
        "repeat": 1,
        "weighted_Q": 1.0381174387489613,
        "all_legal": true,
        "mean_seconds": 0.29970026249854226
      }
    ]
  },
  "terminal": {
    "control": {
      "count": 500,
      "hard_legal": 500,
      "weighted_Q": 1.030132263483138,
      "mean_Q": 1.0476548520922397,
      "p95_Q": 1.1730698440700078,
      "max_Q": 1.6603801169918313,
      "mean_complete_seconds": 0.2351904032001039,
      "p95_complete_seconds": 0.5572573250065029,
      "fallbacks": 0,
      "raw_hard_legal": 0,
      "draft_module_pass10_fraction": 0.47544813349575543,
      "draft_worst_module_coverage_mean": 0.5032190269735521
    },
    "plus": {
      "count": 500,
      "hard_legal": 500,
      "weighted_Q": 1.027018743207681,
      "mean_Q": 1.044722756640298,
      "p95_Q": 1.160293878619024,
      "max_Q": 1.8984315499905384,
      "mean_complete_seconds": 0.228589632800431,
      "p95_complete_seconds": 0.5209298049943754,
      "fallbacks": 0,
      "raw_hard_legal": 0,
      "draft_module_pass10_fraction": 0.478474990471203,
      "draft_worst_module_coverage_mean": 0.5119091106496866
    }
  }
}

训练集反馈决定接纳；验证集只作冻结后的终末评价，不回流。原基底未确认充分训练收敛；保留等更新控制支区分继续拟合与调权收益。测试封存。
