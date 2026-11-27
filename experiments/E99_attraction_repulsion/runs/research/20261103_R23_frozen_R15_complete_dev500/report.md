# Frozen retained AR: fresh complete C3 dev500 acceptance

Weighted neutralQ=1.013885086590, hard legal=500/500.

Complete per-request mean/P95/max seconds: {'mean': 0.21627898059919243, 'p95': 0.5364779150113462, 'max': 2.3119084000063594}; >=250ms: 135/500. Strict acceptance: False.

Timer includes fresh public extraction/cloning/masking/mean-area transform, C3 features/padding/transfers/GPU/output restoration, current retained AR, all original planner/fallback, necessary public-only hard checks and final CUDA sync. Shard disk loading and model/native warmup are separate; no batch amortization. All500 bad/slow cases retained.

One candidate frozen from training before this dev run; no coefficient sweep or checkpoint reselection. Model unchanged, original planner unchanged, test sealed. Current100 four-stage gallery: [open saved gallery](four_stage100/index.html).
