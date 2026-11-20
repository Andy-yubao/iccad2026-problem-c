# E149 initial best52 attached to E148 frozen best46 refiners

Original fixed dev500; only layout passed between models and between units. E148 public-only condition encoder stays unchanged. No training, postprocessing, Q or test.

|Output|Complete Loss|No-overlap Loss|Weighted overlap|
|---|---:|---:|---:|
|initial|0.628453271|0.382388824|0.246064447|
|after_S0x1|0.591173383|0.401117795|0.190055590|
|after_S0x2|0.691523114|0.511072370|0.180450746|
|after_S1x1|0.576663639|0.477035622|0.099628018|
|after_S1x2|0.526730236|0.454403733|0.072326504|
|standard_after_S2x1|0.477071907|0.433426743|0.043645163|
|standard_after_S2x2|0.431891665|0.415743764|0.016147902|
|extended_after_S1x3|0.509323761|0.450233804|0.059089957|
|extended_after_S2x1|0.463569726|0.430959009|0.032610717|
|extended_after_S2x2|0.428690229|0.415324515|0.013365715|
|extended_after_S2x3|0.416349583|0.407027773|0.009321811|

|Comparison|Old final Loss|Attached final Loss|Change|Improved/worsened/unchanged|
|---|---:|---:|---:|---|
|standard_vs_original_E148|0.298647988|0.431891665|+44.616%|4/496/0|
|extended_vs_original_E148_extended|0.294514793|0.416349583|+41.368%|6/494/0|

Initial no-overlap and complete losses reproduce the saved E149 monitoring record. Models are eval/frozen; first64 shared-prefix composition matched a separately composed16-layer trajectory bitwise. No checkpoint or depth reselection.
