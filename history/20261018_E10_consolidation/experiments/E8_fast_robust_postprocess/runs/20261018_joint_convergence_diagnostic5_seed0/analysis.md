# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| joint_shared | 5 | 5 | 5.025407 | 9.999999 | 0.9775 | 1.00 |
| joint_shared_3s | 5 | 5 | 4.997554 | 9.999999 | 1.8289 | 1.00 |
| joint_shared_10s | 5 | 5 | 6.662291 | 9.999999 | 5.6083 | 1.00 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
