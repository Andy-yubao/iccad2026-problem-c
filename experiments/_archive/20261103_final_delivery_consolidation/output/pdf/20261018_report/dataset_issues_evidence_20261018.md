# Dataset evidence | Boundary labels

Fixed local development panel: 100 layouts from the released training pool. All indices below are zero-based; coordinates are XYWH.

- Reference boundary violations: 210 blocks across 91 layouts.
- Near-boundary subset: 97 blocks across 50 layouts, using gaps <=1% of the respective bounding-box side. The official touching tolerance remains 1e-6.
- The following translations were checked individually for nonnegative coordinates and zero pairwise overlap (tolerance 1e-6). They preserve every block's dimensions and all other positions, including preplaced blocks. Whole-solution Q and grouping are not claimed to improve.

1. dev:worker_8/layouts_1232.th:102; block 74; boundary code 6. Before [85.0, 319.0, 23.0, 24.0]; after [85.0, 320.0, 23.0, 24.0]. Required-edge gaps: {'2': 0.0, '4': 1.0}. Maximum normalized gap: 0.291%.

2. dev:worker_29/layouts_3808.th:79; block 96; boundary code 5. Before [0.0, 316.0, 6.0, 7.0]; after [0.0, 317.0, 6.0, 7.0]. Required-edge gaps: {'1': 0.0, '4': 1.0}. Maximum normalized gap: 0.309%.

3. dev:worker_82/layouts_3136.th:40; block 99; boundary code 4. Before [23.0, 306.0, 30.0, 17.0]; after [23.0, 307.0, 30.0, 17.0]. Required-edge gaps: {'4': 1.0}. Maximum normalized gap: 0.309%.

> The translations are separate counterfactuals, not one combined repair. The examples were selected to show small gaps; the aggregate counts above describe the fixed panel, not the whole million-layout dataset.

<!-- pagebreak -->

## Boundary example 1 | Close-up of the one-unit gap

![Boundary close-up](assets/boundary_detail_en.png)

> worker_8/layouts_1232.th:102, block 74. Dashed line: layout top (y=344). The block's top is initially y=343. Translating y=319 to y=320 closes the 1-unit gap; no block changes size, and all other blocks retain their positions. The view is cropped to make the small gap visible; the full layout follows.

<!-- pagebreak -->

## Boundary example 1 | Reference / one-block translation / our saved output

![Boundary example](assets/boundary_1.png)

> worker_8/layouts_1232.th:102, block 74. Right edge already touches; top gap is exactly 1. Middle panel moves y from 319 to 320. All sizes and other positions are unchanged. Unknown metrics show n/a. Red/green boundary badges indicate failed/satisfied requirements.

<!-- pagebreak -->

## MIB incompatibility | Explicit certificate

![MIB permitted area intervals](assets/mib_intervals_en.png)

- Case: worker_0/layouts_1344.th:82. MIB group 1, blocks 33 / 42 / 63.
- Original area intervals: [71.28, 72.72] / [237.60, 242.40] / [55.44, 56.56].
- Conservatively relaxed intervals for round-4 shape buckets: [68.800827, 75.199173] / [235.120827, 244.879173] / [52.960827, 59.039173]. These are still pairwise disjoint: at least 3 shape buckets, hence at least 2 MIB violations.
- The rounding relaxation uses public fixed/preplaced spans to bound rectangle dimensions whenever Q is below the evaluator cap. Capped/infeasible cases are already above the claimed lower bound. This avoids assuming that disjoint exact-area intervals alone prove different rounded shapes at arbitrary aspect ratios.
- Fixed dev100: observed MIB violations 393; certified lower bound 374; arithmetic-mean neutral-runtime Q lower bound 1.180663. Bounds ignore other losses and are not attained-solution claims.

<!-- pagebreak -->

## MIB example | Reference / model proposal / our saved output

![MIB layout](assets/mib_layout.png)

> worker_0/layouts_1344.th:82. MIB group 1 is marked M1 with matching hatching. The reference already contains three distinct shapes in this group. Layouts use the shared visualization/ renderer; metrics come from saved official-evaluator records.

<!-- pagebreak -->

## Reproduction and source record

These figures read existing development JSON records only. No new training, inference, optimization, or sealed-test access was performed. Boundary counterfactuals translate only the identified free block. They demonstrate avoidable boundary violations, without asserting an overall Q improvement or unchanged grouping/connectivity quality.

The published official Q&A (2026-08-27) acknowledges soft violations in reference solutions (A4), gives hard/preplaced priority and the symmetric 1% area rule (A5/A6), and states that hidden cases use the same generation method (A23/A24).

Repository: https://github.com/IntelLabs/FloorSet

Case identifiers are relative to the released FloorSet-Lite training dataset. The development panel is a fixed local selection, not the official public validation set or hidden set. Reported panel statistics must not be generalized to all released data without a broader audit.

The evidence bundle includes layout_evidence.json (reference coordinates, public constraints, area targets, the three one-block translations, and the example MIB certificate), evidence.json (local source paths and SHA-256 hashes), and original-size PNGs. Coordinates alone do not reproduce HPWL or whole-solution Q; those metrics come from the archived evaluator records. Full local sources are listed below.

> history\20261018_E10_consolidation\experiments\intent_aware_repair\runs\20261017_geometry_only_factorial_seed0\evaluation\cases.jsonl

> SHA-256: 7c7a450b4f606d7dbf9a915cadf025884747cbf94b5be6eabd48e365640f45b1

> history\20261018_E10_consolidation\experiments\E8_fast_robust_postprocess\runs\20261018_robust_dev100_seed0\cases.jsonl

> SHA-256: c2d5762767e295a27b3838a029fd20780e9e94509bc4074c0fce1a6a026313e2

> history\20261018_E10_consolidation\experiments\E8_fast_robust_postprocess\runs\20261018_stageA_dev100_seed0\drafts.jsonl

> SHA-256: 33046f8f76f1001ad8e20ffe4c1f2f51c60e5f0fe757126f82712984a94dcf4c

> references\official\results\C_beta_leaderboard_20260823.csv

> SHA-256: f13bfa9ec3b1e972e57e76a692100bc24fd3dd4b1e47fe50b251e96954d6342c

> history\20261018_E10_consolidation\experiments\E9_model_postprocess_alignment\runs\20261018_official_quality_bound_dev100_seed0_r2\results.json

> SHA-256: 8f4d1d070b850d6b59ca5c22e9928d3f9bb5a31e241411ff30fadb59a1b171a8

