# E148 best46: one extra S1 and one extra S2 together

Original fixed dev500. S0x2,S1x3,S2x3:16 geometry layers, frozen original weights. No retraining, postprocessing, Q or test.

|Composition|Final complete Loss|Change vs original|
|---|---:|---:|
|Original12 layers|0.298647988|-|
|Extra S1 only14 layers|0.296352313|-0.769%|
|Extra S2 only14 layers|0.295593039|-1.023%|
|Extra S1 and S2,16 layers|0.294514793|-1.384%|

Final improved/worsened/unchanged vs baseline: 315/185/0.

S1 stage full Loss: 0.363045192 -> 0.346049652.
S2 stage complete Loss after two calls (layer14): 0.296352313; third-call first sublayer15: 0.305210294; complete layer16: 0.294514793.

Exact cached14-layer prefix was reproduced bitwise against direct composition on first64 cases. All500 prefix Loss values matched previous record exactly; original weights/source identities unchanged.
