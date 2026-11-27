# E59 small paired Loss comparison

Numeric recommendation: original; visual inspection pending.

| Common metric | Original mean | Centerline mean | Relative change |
|---|---:|---:|---:|
| overlap_area_fraction | 0.0878804255 | 0.09573404 | +8.94% |
| overlap_pairs | 80.295 | 80.5775 | +0.35% |
| penetration_squared_per_block | 0.0333804498 | 0.0379567107 | +13.71% |
| loss_overlap_original | 0.06895732 | 0.0782451669 | +13.47% |
| loss_overlap_centerline | 0.0995528331 | 0.111465201 | +11.97% |
| loss_center_local | 0.222203942 | 0.218865264 | -1.50% |
| loss_shape | 0.201442352 | 0.199814967 | -0.81% |
| loss_relative_local | 0.273251013 | 0.268184923 | -1.85% |
| loss_contact | 0.116539749 | 0.112968439 | -3.06% |
| loss_boundary | 0.0182828681 | 0.0192260078 | +5.16% |
| loss_mib | 0.0126446281 | 0.0119462729 | -5.52% |
| loss_bbox | 0.000411631463 | 0.00026978841 | -34.46% |

Failed preregistered gates: overlap_mean_down5pct, paired_cluster_ci_below_zero, overlap_p95_within2pct, overlap_worst5_within2pct, loss_boundary_guard.

Each candidate is its own fixed-Loss best. Cross-objective total Loss is not used to rank definitions.
Development comparisons and a single seed do not establish independent generalization or universal superiority.
See paired_comparison.json for paired shard/case bootstrap intervals, tails, strata and five existing visual pairs.
