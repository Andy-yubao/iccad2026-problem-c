# E140 early G/B x4 versus E139

One changed factor: S2/S6 grouping and boundary coefficients x4. E139 best42 reused without retraining.

|Model|Mixed Q|Hard legal|Grouping violations|Boundary violations|All soft satisfied|Mean/P95 ms|Common final Loss|
|---|---:|---:|---:|---:|---:|---:|---:|
|E139_existing_best42|1.016845|500/500|83|54|384/500|173.3/406.4|0.354014|
|e119_front_gb4|1.039537|500/500|106|82|378/500|197.1/481.8|0.341736|
