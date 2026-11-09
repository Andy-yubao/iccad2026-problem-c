# E59 provisional Loss selection

Use the original overlap definition for the large-data model. The calibrated centerline_v1 candidate failed the predeclared dev2000 geometry gates. This is a choice between these two fixed training configurations, not a proof that every centerline formulation is inferior.

Pairwise intersection sum divided by total public area: 0.0878804 -> 0.0957340 (+8.94%); P95 +10.45%, worst5% mean +9.96%. Only471/2000 cases improve on overlap area. The paired source-shard bootstrap 95% interval for the absolute mean difference is [0.0072057, 0.0084894], wholly positive. Center, shape and relative-reference components improve modestly, while boundary worsens5.16%.

Both terminal panels are500/500 hard legal. Flash weighted Q is1.057557 versus1.055286 (-0.215%); this does not offset raw draft-geometry regression and is below the execution variation observed for identical control-model weights. No checkpoint is reselected or re-evaluated.

Four actual reference/current/E42 image pairs were viewed, covering an improvement, a regression, a typical difference and a worst-overlap baseline case. These are AI observations, not user scores:

- case0115: Centerline reduces the top-block7 collision and changes the yellow-group placement toward reference; several central and lower contacts still overlap. A genuine per-case improvement exists; no claim of dominance.

- case0225: Both drafts preserve broad reference regions, but centerline has more concentrated intersections around the lower G1/G3 region and the mid-left fixed blocks. Supports inspecting intersection severity rather than red-outline count alone.

- case0355: Macro regions remain similar. Centerline relocates blocks26/53 and changes upper22/63 relations while many collisions persist elsewhere; there is no consistent overall visual advantage.

- case0449: Centerline separates the nearly coincident top blocks0/15, a useful local improvement; the MIB49/nearby blocks and lower region remain crowded. This counterexample prevents claiming it fails on every case.

Continue with frozen original fixed-G, graph-prenormalized160x12, train64000, seed0, batch32, FP32, unchanged dev500/LR/early-stop, at most4hours. This jointly changes training data and width compared with the small study and does not isolate their separate effects.
