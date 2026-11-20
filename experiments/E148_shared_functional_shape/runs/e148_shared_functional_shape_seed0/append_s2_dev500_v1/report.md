# E148 frozen best46: append one trained S2 unit

Original fixed dev500. Original12 layers versus14 layers. No retraining, postprocessing, Q or test reading.

|Output|Complete final Loss|
|---|---:|
|Original layer12|0.298647988|
|Appended S2, first sublayer13|0.306707202|
|Appended S2, complete layer14|0.295593039|

Change: -0.003054949 (-1.023%). Improved/worsened/unchanged: 358/142/0.

Same full base Loss and coefficients, per-case arithmetic mean. Original baseline reproduced within2e-7; checkpoint/model parameters unchanged.

|Weighted component|Layer12|Layer14|Change|
|---|---:|---:|---:|
|reference_weighted|0.227447864|0.229361971|+0.001914107|
|overlap_weighted|0.014986737|0.010015850|-0.004970886|
|wire_compact_weighted|0.053103684|0.053051193|-0.000052491|
|grouping_weighted|0.001420593|0.001208556|-0.000212038|
|boundary_weighted|0.000225976|0.000218032|-0.000007944|
|public_shape_weighted|0.000562581|0.000688674|+0.000126093|
|public_preplaced_position_weighted|0.000900553|0.001048762|+0.000148210|
