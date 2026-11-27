# Repair stability review

Completed fixed-G dev2000: 0.357134791; health audit passed.

The final hidden maximum is 188.886, and final SiLU exact-zero fraction is 0.0001402 on the same five diagnostic training cases. Original seed1 had hidden maximum 53,390,224 and zero fraction 1.0. The full run completed 27,000 updates without triggering the numerical guard. This is evidence of a successful repair trajectory, not a universal stability guarantee.

Ordering audit: the same seed1 and same 16,000 training members were used, but the original E57 subset-index list was permuted whereas E59 repair used natural registry order. This changed actual minibatch ordering. Together with the documented staged engine, this prevents treating the completed-run Loss improvement as an exact one-factor causal ablation. The old fixed-weight, same-case graph-branch interventions independently support the amplification-path diagnosis. Historical config/source records are preserved unchanged.

The small original-vs-centerline comparison uses exactly the same 4096 index order and paired seed in both arms. Its original-control model tensors were verified bit-identical to E57. Its flash Q changed from 1.041537 to 1.057557 despite identical learned tensors; this is execution/postprocessor sensitivity and must not be claimed as a learned-model difference. Do not rerun the terminal panel to select a preferred Q.
