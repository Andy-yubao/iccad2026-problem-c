"""Preserve raw topology and use hard separation; construct only for recovery."""
import time
import numpy as np
from ...planner.backend.frozen.packing import prepare_problem,_dimensions
from ...planner.backend.frozen.feasible import Objective,row_control
from ...planner.backend.frozen.solver import hard_valid
from ...planner.backend.guard import repair_tiny_negative
from ...planner.fast_geometry import topology,reduce_edges,contact_tree,step
from ...planner.projection import solve as project
from ...planner.canonical import canonicalize
from ...planner.certify_polish import polish
from ...planner.legal_seed import seed
from .normalization import shrink


def optimize(draft,inputs,settings,organized=False):
    start=time.perf_counter();deadline=start+settings['seconds']
    p=prepare_problem(*inputs);obj=Objective(p);fallback=row_control(p);best=fallback.copy();value=obj(best);trace=[];useful=False
    state=np.array(draft,copy=True);w,h=_dimensions(p,state[:,2]/state[:,3],True);state[:,2:]=np.column_stack((w,h));state[p.preplaced]=p.specs[p.preplaced]
    prefix=state.copy()
    def accept(candidate):
        nonlocal best,value,useful
        if candidate is None:return None
        candidate=canonicalize(candidate,p,tolerance=1e-4);candidate,_=repair_tiny_negative(candidate,p)
        if not hard_valid(candidate,p):return None
        score=obj(candidate)
        if score<value:best=candidate.copy();value=score;useful=True
        return candidate
    accept(draft);accept(state);route_failed=False;seeded=False
    for iteration in range(settings['maximum_shape_solves']):
        remaining=deadline-time.perf_counter()-settings['projection_and_polish_reserve_seconds']
        if remaining<.018:break
        if iteration==2 and not useful:
            t=time.perf_counter();constructed=seed(p,prefix,0,preserve_centers=True);clean=accept(constructed)
            trace.append(dict(phase='lazy_legal_seed',seconds=time.perf_counter()-t,legal=clean is not None));seeded=True
            if clean is not None:state=clean
            remaining=deadline-time.perf_counter()-settings['projection_and_polish_reserve_seconds']
            if remaining<.018:break
        route=False
        full=topology(state,p.constraints,mode=int(route),array=True);edges=reduce_edges(full,p.n)
        proposal,stats=step(p,state,edges,seconds=min(settings['shape_cap_seconds'],remaining),full_edges=full,conservative=not organized,elastic=0.,reweighted=True)
        if organized and proposal is not None:
            original_wh=proposal[:,2:].copy();proposal=shrink(proposal,p)
            assert (proposal[:,2:]<=original_wh+1e-12).all()
        previous=value;clean=accept(proposal);trace.append(dict(phase='shape',route=route,legal=clean is not None,**stats))
        if clean is not None:state=clean
        else:
            route_failed=route_failed or route
            if useful:state=best.copy()
            if seeded:break
        if iteration>=1 and useful and previous-value<1e-5 and sum(obj.violations(best))==0:break
    if not useful:
        # Cheap constructive recovery is needed even when a shape solve consumed
        # its entire limit; this branch is included in total timing and evidence.
        t=time.perf_counter();clean=accept(seed(p,prefix,0,preserve_centers=True));seeded=True
        trace.append(dict(phase='terminal_legal_seed',seconds=time.perf_counter()-t,legal=clean is not None))
    remaining=deadline-time.perf_counter()-.008
    if remaining>.008:
        state=best.copy();full=topology(state,p.constraints,mode=0,array=True)
        proposal,stats=project(p,state,reduce_edges(full,p.n),contact_tree(state,p,full),seconds=remaining,quadratic=True)
        clean=accept(proposal);trace.append(dict(phase='project',legal=clean is not None,**stats))
    remaining=deadline-time.perf_counter()
    if remaining>.001:
        best,stats=polish(best,p,seconds=min(.008,remaining),max_accept=4,push=True)
        trace.append(dict(phase='certified_push',**stats))
    assert hard_valid(best,p)
    return best,dict(seconds=time.perf_counter()-start,trace=trace,row_retained=bool(np.array_equal(best,fallback)),constructive_recovery=seeded,proxy=obj(best)),prefix
