# E148 best46: L4 versus final Loss association

Fixed original dev500,500 paired cases from299 source shards. Frozen saved per-case records only; no inference, training, postprocessing, Q or test.

|Early measure|Final output|Pearson r|Cluster95% interval|Spearman rho|Cluster95% interval|Final worst50 also early worst50|Final worst50 also early worst100|
|---|---|---:|---|---:|---|---:|---:|
|L4_full|L12_full|0.940135|[0.924189, 0.953273]|0.927925|[0.910802, 0.940011]|42/50|49/50|
|L4_full|L16_full|0.933602|[0.915378, 0.948565]|0.917677|[0.897868, 0.931470]|41/50|49/50|
|L4_stage_no_overlap|L12_full|0.968251|[0.958021, 0.975525]|0.954974|[0.942507, 0.963353]|47/50|50/50|
|L4_stage_no_overlap|L16_full|0.962649|[0.950239, 0.971167]|0.944949|[0.929769, 0.955219]|45/50|50/50|

L4 full includes direct overlap; L4 stage objective excludes it. Both use original coefficients, no auxiliary layer weight. Final uses complete single-output Loss.

Source-shard cluster bootstrap:2000 repeats, seed20261117. For independent random rankings, worst50 overlap would average5 cases.

|L4 component|Pearson vs L12|Spearman vs L12|Pearson vs L16|Spearman vs L16|
|---|---:|---:|---:|---:|
|reference_weighted|0.971694|0.959820|0.966528|0.950324|
|overlap_weighted|0.726333|0.741127|0.719222|0.731816|
|wire_compact_weighted|-0.362383|-0.329258|-0.363077|-0.329177|
|grouping_weighted|0.341716|0.388166|0.336155|0.380867|
|boundary_weighted|0.432847|0.398982|0.426913|0.390554|
|public_shape_weighted|0.003147|0.021669|0.003512|0.022583|
|public_preplaced_position_weighted|-0.023681|-0.011537|-0.026707|-0.010973|

After removing the weighted reference component from both full Loss values:
- L4 vs L12: Pearson 0.622044, Spearman 0.618772.
- L4 vs L16: Pearson 0.502717, Spearman 0.491512.

These are associations on the observed panel, not proof of a causal early-layout bottleneck or of insufficient downstream correction. Shared reference fitting and common problem difficulty can contribute.
