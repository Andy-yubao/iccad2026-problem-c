# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| conic_quadratic | 5 | 5 | 3.130601 | 9.999999 | 0.8834 | 1.00 |
| conic_quadratic_relaxed | 5 | 5 | 1.422020 | 1.507053 | 0.8677 | 1.00 |
| conic_quadratic_3s | 5 | 5 | 1.363946 | 1.491825 | 1.5364 | 1.00 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
