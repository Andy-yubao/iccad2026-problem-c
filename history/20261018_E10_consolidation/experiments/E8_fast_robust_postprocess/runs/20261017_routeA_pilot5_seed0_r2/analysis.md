# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 5 | 5 | 1.157900 | 1.209775 | 14.9751 | 1.00 |
| control_short | 5 | 5 | 8.028286 | 9.999999 | 1.5493 | 9.67 |
| sparse_short | 5 | 5 | 9.999999 | 9.999999 | 1.4833 | 10.10 |
| dense_short | 5 | 5 | 8.240024 | 9.999999 | 1.4918 | 10.04 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
