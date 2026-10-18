# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| joint_independent | 5 | 5 | 8.434377 | 9.999999 | 0.4787 | 1.00 |
| joint_shared | 5 | 5 | 8.433995 | 9.999999 | 0.4766 | 1.00 |
| joint_fixed | 5 | 5 | 8.441142 | 9.999999 | 0.4877 | 1.00 |
| joint_shared_3s | 5 | 5 | 8.433995 | 9.999999 | 0.4869 | 1.00 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
