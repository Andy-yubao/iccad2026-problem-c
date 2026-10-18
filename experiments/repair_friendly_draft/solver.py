"""Public-input checked entry point for the frozen refinement mechanism.

The core benchmark is unchanged. This final guard addresses the negative-zero
counterexample found in the unweighted ablation; it changes none of the100
successful core benchmark outputs. No ground-truth/reference inputs accepted.
"""
import time
import numpy as np
from .global_refine import refine
from .packing import prepare_problem
from .postprocess import finalize
from .feasible import row_control


def hard_valid(pos,p):
    a=np.asarray(pos,dtype=float)
    if a.shape!=(p.n,4) or not np.isfinite(a).all() or (a[:,2:]<=0).any() or (a[:,:2]<0).any():return False
    locked=p.fixed|p.preplaced
    if (np.abs(a[locked,2:]-p.specs[locked,2:])>1e-8).any():return False
    if (np.abs(a[p.preplaced]-p.specs[p.preplaced])>1e-8).any():return False
    free=~locked
    if (np.abs(a[free,2:].prod(1)-p.areas[free])/p.areas[free]>.01).any():return False
    i,j=np.triu_indices(p.n,1)
    overlap=np.minimum(a[i,:2]+a[i,2:],a[j,:2]+a[j,2:])-np.maximum(a[i,:2],a[j,:2])
    return not (overlap>1e-6).all(1).any()


def optimize(raw,inputs,seconds=120.,shape_steps=24,reweighted=True,**options):
    begin=time.monotonic();p=prepare_problem(*inputs)
    candidate,stats=refine(raw,inputs,seconds=seconds,shape_steps=shape_steps,reweighted=reweighted,**options)
    rejected=not hard_valid(candidate,p)
    if rejected:
        candidate,_=finalize(raw,inputs,2.)
        if not hard_valid(candidate,p):candidate=row_control(p)
    if not hard_valid(candidate,p):raise RuntimeError('No hard-legal candidate after fallback')
    stats.update(guard_fallback=rejected,checked_total_seconds=time.monotonic()-begin)
    return candidate,stats
