# E148 sparse connectivity check

Frozen best46, original fixed dev500. True initial generator checked once without updating weights; L4/L12/L16 losses reused. No Q/postprocessing/test.

|Outcome|Density Spearman|Partial rank: N|Partial rank: public controls|Cluster95% public-adjusted|
|---|---:|---:|---:|---|
|initial_full|0.023122|0.032744|0.026845|[-0.09046234163495448, 0.1445649871147032]|
|initial_reference|0.025777|0.033325|0.043062|-|
|initial_overlap|0.016640|0.025682|0.006282|-|
|L4_full|-0.403422|-0.406252|-0.648266|[-0.703499942339736, -0.5792742758281365]|
|L4_reference|-0.435100|-0.435056|-0.700573|-|
|L4_overlap|-0.333613|-0.344846|-0.453437|-|
|L12_full|-0.461191|-0.461032|-0.684419|[-0.7382789495456958, -0.6161323670459654]|
|L16_full|-0.461559|-0.461428|-0.674430|[-0.7275427060508423, -0.6064998430941377]|
|L16_reference|-0.473869|-0.473694|-0.682034|-|
|L16_overlap|-0.468840|-0.468570|-0.625228|-|
|L16_without_reference|-0.030868|-0.030777|-0.059066|-|

Negative coefficient means denser graph is associated with lower Loss.

|Group|Count|Mean N|Mean degree|Mean density|Isolated fraction|Initial full|L4 full|L16 full|L16 reference|L16 without reference|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|sparsest125|125|69.56|7.349|0.10778|0.007|5.065614|0.627035|0.372753|0.307224|0.065529|
|densest125|125|67.61|31.258|0.46666|0.000|5.630441|0.437936|0.205630|0.141529|0.064101|

Public controls: N, F-or-P fraction, P fraction, pin-connected fraction, group-member fraction, boundary fraction, area CV. Cluster bootstrap respects299 source shards;1000 repeats.

Exact indistinguishable isolated/free peers: {'cases_with_exactly_indistinguishable_free_peers': 0, 'groups': 0, 'max_initial_state_difference': None}
Density vs final, additionally adjusted for L4: -0.3078778148082988

One frozen model and reused dev500; observational correlation/conditioning does not establish causality. Initial=actual initial generator, L4=two S0 calls.
