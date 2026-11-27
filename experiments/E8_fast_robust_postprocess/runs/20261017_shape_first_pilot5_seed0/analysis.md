# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| control120 | 5 | 5 | 1.157900 | 1.209775 | 14.9751 | 1.00 |
| shape_first | 5 | 5 | 6.524206 | 9.999999 | 1.3732 | 10.91 |
| shape_first_3s | 5 | 5 | 4.768720 | 9.999999 | 2.6946 | 5.56 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
