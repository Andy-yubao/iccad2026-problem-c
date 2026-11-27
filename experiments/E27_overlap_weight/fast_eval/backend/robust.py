"""Fast conic path with bounded, public-status-triggered feasible-seed rescue."""
import time
import numpy as np
from .conic_geometry import optimize as geometry
from .frozen.postprocess import finalize
from .frozen.packing import prepare_problem
from .frozen.feasible import Objective
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .guard import repair_tiny_negative


def fast(draft,inputs,seconds):
    return geometry(draft,inputs,seconds=seconds,conservative=True,
                    reduced=True,quadratic_wire=True,diagonal_wire=True)


def optimize(draft,inputs,seconds=5.,fast_seconds=1.3):
    begin=time.perf_counter();deadline=begin+seconds
    best,stats=fast(draft,inputs,min(fast_seconds,seconds))
    stats['rescue_used']=False
    # Events only record strict public-proxy improvements over the legal row.
    # No reference score, layout ID, or outcome threshold enters this gate.
    if stats['improvements'] or deadline-time.perf_counter()<.15:
        stats['checked_total_seconds']=time.perf_counter()-begin
        return best,stats
    stats['rescue_used']=True;stats['rescue_reason']='no_candidate_improved_initial_legal_row'
    p=prepare_problem(*inputs);objective=Objective(p);value=objective(best)
    remaining=deadline-time.perf_counter()
    start=time.perf_counter()
    packed,packing=finalize(draft,inputs,seconds=min(1.2,remaining*.35))
    packed=canonicalize(packed,p);packed,_=repair_tiny_negative(packed,p)
    valid=hard_valid(packed,p)
    stats['rescue_packing']={**packing,'strict_valid':valid,'seconds':time.perf_counter()-start}
    if valid:
        score=objective(packed)
        if score<value:
            best=packed.copy();value=score
            stats['improvements'].append({'elapsed':time.perf_counter()-begin,'proxy':float(value),
                                          'phase':'feasible_seed_rescue','positions':best.tolist()})
    remaining=deadline-time.perf_counter()-.04
    if valid and remaining>.1:
        offset=time.perf_counter()-begin
        proposed,rescue=fast(packed,inputs,remaining)
        events=rescue.pop('improvements',[])
        for event in events:
            if event['proxy']<value:
                # Preserve a single monotone public-proxy improvement curve.
                stats['improvements'].append({**event,'elapsed':offset+event['elapsed'],
                                              'phase':'rescue_'+event['phase']})
                value=event['proxy']
        if objective(proposed)<objective(best):best=proposed
        stats['rescue_refinement']=rescue
    if not hard_valid(best,p):raise AssertionError('Rescue returned invalid incumbent')
    stats['proxy']=float(objective(best));stats['seconds']=time.perf_counter()-begin
    stats['checked_total_seconds']=stats['seconds']
    stats['initial_row_retained']=not bool(stats['improvements'])
    return best,stats
