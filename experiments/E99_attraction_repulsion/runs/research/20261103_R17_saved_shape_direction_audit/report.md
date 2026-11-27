# Saved-operation rectangle shape-direction audit

Read-only train500 trajectories; no new layout, planner solve or test access. Finite differences check only the contact derivative with direction frozen.

| Source | Pair contributions with opposite sign | Force-weighted opposite fraction | Aggregated module opposite fraction |
|---|---:|---:|---:|
| R15/budget_gate | 7.501% | 10.971% | 16.485% |
| R15/attraction_equal_slope | 7.606% | 10.569% | 20.096% |
| R16/longer_relaxation | 7.417% | 10.493% | 21.225% |

An active-face derivative can grow large near oblique thin rectangles. Existing shape and movement limits remain necessary. A mismatch does not establish a quality gain; a future shape-law arm requires matched final-Q controls with the original planner.
