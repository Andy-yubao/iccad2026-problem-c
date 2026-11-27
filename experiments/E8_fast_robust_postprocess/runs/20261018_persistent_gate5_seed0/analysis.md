# E8 paired postprocessor results

Status: complete; complete 5/5.

Q uses official neutral runtime; all failures count as Q=10. Pilot5 is not a dev100 or generalization conclusion.

| Variant | n | feasible | Q mean | Q max | post mean s | post speedup |
|---|---:|---:|---:|---:|---:|---:|
| safe_plain120 | 5 | 5 | 1.363557 | 1.491825 | 8.4798 | 1.00 |
| conic_diagonal | 5 | 5 | 1.350556 | 1.472866 | 0.8833 | 9.60 |

Subprocess startup is excluded from postprocessing and included separately in worker wall time. Observational wrapper overhead is included in controls; final speed claims require a plain-control timing check.
