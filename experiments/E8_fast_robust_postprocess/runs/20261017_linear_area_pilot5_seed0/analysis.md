# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 5 | 5 | 1.157900 | 1.209775 | 14.9751 | 1.00 |
| linear_area | 5 | 5 | 8.287332 | 9.999999 | 1.2904 | 11.60 |
| linear_area_3s | 5 | 5 | 8.272656 | 9.999999 | 2.4567 | 6.10 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
