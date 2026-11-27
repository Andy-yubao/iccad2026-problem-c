# E8 paired postprocessor results

Status: user_stop; complete 8/100.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| safe_plain120 | 9 | 8 | 2.214867 | 10.000000 | 8.1600 | 1.00 |
| conic_diagonal | 8 | 8 | 1.264799 | 1.384836 | 1.2481 | 5.27 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
