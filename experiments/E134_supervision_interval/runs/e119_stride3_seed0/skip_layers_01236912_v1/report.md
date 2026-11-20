# Frozen layer skipping: E134 stride3 best63

Keep S0 frontend and original geometry blocks 1,2,3,6,9,12. All weights unchanged. Same dev500, no training/postprocessing.

|Output|Original mean Loss|Skipped mean Loss|Delta|Relative change|Improved/worsened/unchanged|
|---|---:|---:|---:|---:|---:|
|S0|4.938188|4.938188|+0.000000|+0.00%|0/0/500|
|S1|1.570059|1.570059|+0.000000|+0.00%|0/0/500|
|S2|0.893110|0.893110|+0.000000|+0.00%|0/0/500|
|S3|0.540194|0.540194|+0.000000|+0.00%|0/0/500|
|S6|0.442186|0.541746|+0.099560|+22.52%|2/498/0|
|S9|0.383794|0.545031|+0.161237|+42.01%|0/500/0|
|S12|0.334363|0.546144|+0.211782|+63.34%|0/500/0|
|original_weighted_objective|0.358568|0.545380|+0.186812|+52.10%|0/500/0|

Final component values and source-shard paired confidence intervals: `results.json`. Per-case comparisons: `per_case.csv`.

S0-S3 states and Loss are bitwise identical to the original capture; checkpoint and original sources verified unchanged.
