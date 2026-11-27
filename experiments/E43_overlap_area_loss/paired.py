"""Saved-case paired uncertainty; no model/solver execution."""
import numpy as np


def paired(candidate, control):
    q = np.array([a['metrics']['cost']-b['metrics']['cost'] for a,b in zip(candidate,control)])
    n = np.array([r['n'] for r in control]); weights = np.exp((n-n.max())/12)
    groups = {}
    for i,r in enumerate(control):
        groups.setdefault(r['member'][0].split('/')[0], []).append(i)
    blocks = list(groups.values()); rng = np.random.default_rng(43)
    cases, workers = [], []
    for _ in range(4000):
        ids = rng.integers(len(q),size=len(q))
        cases.append(np.average(q[ids],weights=weights[ids]))
        ids = np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        workers.append(np.average(q[ids],weights=weights[ids]))
    return {'weighted_Q_delta':float(np.average(q,weights=weights)),
        'case_bootstrap_95':np.percentile(cases,[2.5,97.5]).tolist(),
        'worker_cluster_bootstrap_95':np.percentile(workers,[2.5,97.5]).tolist(),
        'Q_better_equal_worse':[int((q<-1e-9).sum()),int((abs(q)<=1e-9).sum()),int((q>1e-9).sum())],
        'limitation':'One seed and one timed repair run; intervals do not cover training seeds or inter-run timing.'}
