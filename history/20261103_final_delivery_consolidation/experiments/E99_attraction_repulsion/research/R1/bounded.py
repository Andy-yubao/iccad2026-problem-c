"""Bounded conic depth and a saved hard-legal incumbent, shared by R1 arms."""
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
from .field import organize


def optimize(draft,inputs,settings,organized=False):
    start=time.perf_counter();deadline=start+.180
    p=prepare_problem(*inputs);obj=Objective(p);fallback=row_control(p)
    best=fallback.copy();value=obj(best);trace=[]
    state=np.array(draft,copy=True)
    w,h=_dimensions(p,state[:,2]/state[:,3],True)
    centers=state[:,:2]+state[:,2:]/2
    state[:,2:]=np.column_stack((w,h));state[:,:2]=np.maximum(0,centers-state[:,2:]/2);state[p.preplaced]=p.specs[p.preplaced]
    if organized:
        state,stats=organize(draft,p,settings);trace.append(dict(phase='synchronous_field',**stats))
    prefix=state.copy()
    def accept(candidate):
        nonlocal best,value
        if candidate is None:return None
        candidate=canonicalize(candidate,p,tolerance=1e-4);candidate,_=repair_tiny_negative(candidate,p)
        if not hard_valid(candidate,p):return None
        score=obj(candidate)
        if score<value:best=candidate.copy();value=score
        return candidate
    accept(draft);accept(state)
    for iteration in range(4):
        remaining=deadline-time.perf_counter()-.033
        if remaining<.018:break
        full=topology(state,p.constraints,mode=1,array=True,group_order=True)
        edges=reduce_edges(full,p.n)
        proposal,stats=step(p,state,edges,seconds=min(.045,remaining),full_edges=full,
            conservative=True,elastic=100000.*(1+iteration/3),reweighted=True)
        clean=accept(proposal);trace.append(dict(phase='shape',legal=clean is not None,**stats))
        if proposal is None:break
        state=clean if clean is not None else proposal
    remaining=deadline-time.perf_counter()-.008
    if remaining>.008:
        if not np.array_equal(best,fallback):state=best.copy()
        full=topology(state,p.constraints,mode=1,array=True,group_order=True)
        proposal,stats=project(p,state,reduce_edges(full,p.n),contact_tree(state,p,full),seconds=remaining,quadratic=True)
        clean=accept(proposal);trace.append(dict(phase='project',legal=clean is not None,**stats))
    remaining=deadline-time.perf_counter()
    if remaining>.001:
        best,stats=polish(best,p,seconds=min(.008,remaining),max_accept=4,push=True)
        trace.append(dict(phase='certified_push',**stats))
    assert hard_valid(best,p)
    return best,dict(seconds=time.perf_counter()-start,trace=trace,row_retained=bool(np.array_equal(best,fallback)),proxy=obj(best)),prefix
