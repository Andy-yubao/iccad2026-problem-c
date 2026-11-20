# E148 best46: insert one S0 or one S1 separately

Original fixed dev500, frozen original checkpoint; each additional unit has two geometry sublayers. All downstream units are rerun. No training/postprocessing/Q/test.

S0 stage objective excludes the entire direct overlap group exactly as original supervision; full Loss includes it. S1/S2 stage objectives are full Loss. No cross-layer weighting here.

|Insertion|Stage objective before|Stage objective after|Full stage Loss before|Full stage Loss after|Final full Loss|Final relative change|
|---|---:|---:|---:|---:|---:|---:|
|insert_S0|0.347111906|0.390704447|0.538000163|0.567440094|0.337401470|+12.976%|
|insert_S1|0.363045192|0.346049652|0.363045192|0.346049652|0.296352313|-0.769%|

Original final full Loss: 0.298647988. Prior appended S2 final: 0.295593039 (-1.023%), reused without rerunning.

|Arm|S0 stage full Loss|S1 stage full Loss|S2 final full Loss|Final improved/worsened/unchanged|
|---|---:|---:|---:|---|
|baseline|0.538000163|0.363045192|0.298647988|-|
|insert_S0|0.567440094|0.410521364|0.337401470|44/456/0|
|insert_S1|0.538000163|0.346049652|0.296352313|286/214/0|

|Weighted final component|Extra S0 change|Extra S1 change|
|---|---:|---:|
|reference_weighted|+0.012197779|+0.000542731|
|overlap_weighted|-0.000269350|-0.002521117|
|wire_compact_weighted|+0.000800993|-0.000185437|
|grouping_weighted|+0.000066803|-0.000162169|
|boundary_weighted|+0.000033167|-0.000024502|
|public_shape_weighted|+0.007531077|+0.000031327|
|public_preplaced_position_weighted|+0.018393012|+0.000023494|
