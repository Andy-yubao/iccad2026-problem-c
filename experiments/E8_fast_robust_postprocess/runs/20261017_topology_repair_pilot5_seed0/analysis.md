# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 5 | 5 | 1.157900 | 1.209775 | 14.9751 | 1.00 |
| topology_short | 5 | 5 | 6.832055 | 9.999999 | 0.6215 | 24.10 |
| topology_3s | 5 | 5 | 5.423926 | 9.999999 | 0.9151 | 16.36 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
