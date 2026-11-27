# E8 paired postprocessor results

Status: complete; complete 100/100.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 100 | 100 | 1.276534 | 5.379855 | 10.4664 | 1.00 |
| conic_diagonal | 100 | 100 | 1.211556 | 1.472866 | 1.1882 | 8.81 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
