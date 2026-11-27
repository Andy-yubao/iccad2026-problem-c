# E8 paired postprocessor results

Status: complete; complete 2/2.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| conic_diagonal_3s | 2 | 2 | 5.897778 | 9.999999 | 1.6967 | 1.00 |
| conic_diagonal_relaxed_3s | 2 | 2 | 5.831582 | 9.999999 | 1.7298 | 1.00 |
| conic_diagonal_10s | 2 | 2 | 5.897778 | 9.999999 | 2.1934 | 1.00 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
