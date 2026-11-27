# E8 paired postprocessor results

Status: complete; complete 100/100.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| safe_plain120 | 100 | 100 | 1.280117 | 3.353213 | 12.7105 | 1.00 |
| conic_diagonal | 100 | 100 | 1.311713 | 9.999999 | 1.1993 | 10.60 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
