# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| conic_short | 5 | 5 | 3.130265 | 9.999999 | 1.1187 | 1.00 |
| conic_3s | 5 | 5 | 1.391586 | 1.491825 | 1.9021 | 1.00 |
| conic_relaxed | 5 | 5 | 1.426480 | 1.513724 | 1.0935 | 1.00 |
| conic_dense | 5 | 5 | 3.130265 | 9.999999 | 1.1399 | 1.00 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
