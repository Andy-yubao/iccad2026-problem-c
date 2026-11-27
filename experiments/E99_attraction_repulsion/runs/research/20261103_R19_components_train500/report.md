# Center movement and strain ablation: train500

Both mechanisms controlQ=1.010848448528; noAR=1.013901252087.

| Arm | WeightedQ | Hard legal | CPU mean ms | CPU max ms |
|---|---:|---:|---:|---:|
| translation_only | 1.012081446344 | 500/500 | 201.973 | 3179.420 |
| shape_only | 1.013047974789 | 500/500 | 204.965 | 3154.886 |

Weighted interaction (both - translation - shape + neither)=-0.000379720519; this is an algorithm-pipeline interaction on adaptive training sources, not a physical or population causal estimate.

Only disable one AR mechanism per arm. Original planner/config unchanged, all500 members retained. Ten identical preflight records reused,990 fresh full solves. CPU cached-draft timing excludes fresh C3; no complete250ms or dev claim; test sealed.
