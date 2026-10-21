"""Repair tiny negative roundoff, then enforce the original strict hard guard."""
import numpy as np
from .frozen.solver import hard_valid
from .frozen.packing import prepare_problem


def repair_tiny_negative(candidate, problem, tolerance=1e-9):
    original=np.asarray(candidate,dtype=float)
    stats={'attempted':False,'applied':False,'coordinate_count':0,'max_abs_change':0.}
    if original.shape!=(problem.n,4) or not np.isfinite(original).all():
        stats['reason']='invalid_array'
        return original,stats
    mask=(original[:,:2]<0)&(original[:,:2]>=-tolerance)
    mask[problem.preplaced]=False
    if not mask.any():
        stats['reason']='no_tiny_negative_movable_coordinate'
        return original,stats
    proposed=original.copy()
    stats.update(attempted=True,coordinate_count=int(mask.sum()),
                 max_abs_change=float(np.abs(original[:,:2][mask]).max()))
    proposed[:,:2][mask]=0.
    if hard_valid(proposed,problem):
        stats.update(applied=True,reason='all_original_strict_hard_checks_pass')
        return proposed,stats
    stats['reason']='full_hard_guard_rejected_repair'
    return original,stats


def guarded_refine(refine):
    """Wrap only the final candidate; keep the entire search trajectory intact."""
    def call(raw,inputs,**kwargs):
        candidate,stats=refine(raw,inputs,**kwargs)
        candidate,repair=repair_tiny_negative(candidate,prepare_problem(*inputs))
        stats={**stats,'tiny_negative_cleanup':repair}
        return candidate,stats
    return call
