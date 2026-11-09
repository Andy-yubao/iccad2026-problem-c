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
from ..R6.native_seed import seed
from ..R9.obstacles import seed as obstacle_seed
from .shape import step as public_shape_step
from ..R4.normalization import shrink


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
    accept(draft);accept(state);route_failed=False;seeded=False;overrides={};preconditioned=False;expanded=False
    for iteration in range(settings['maximum_shape_solves']):
        remaining=deadline-time.perf_counter()-settings['projection_and_polish_reserve_seconds']
        if remaining<.018:break
        if iteration==2 and not useful and not organized:
            t=time.perf_counter();constructed=seed(p,prefix,0,preserve_centers=True);clean=accept(constructed)
            trace.append(dict(phase='lazy_legal_seed',seconds=time.perf_counter()-t,legal=clean is not None));seeded=True
            if clean is not None:state=clean
            remaining=deadline-time.perf_counter()-settings['projection_and_polish_reserve_seconds']
            if remaining<.018:break
        route=False
        full=topology(state,p.constraints,mode=0,array=True)
        for (i,j),edge in overrides.items():full[max(i,j)*(max(i,j)-1)//2+min(i,j)]=edge
        edges=reduce_edges(full,p.n)
        proposal,stats=public_shape_step(p,state,edges,public_domain=expanded,seconds=min(settings['shape_cap_seconds'],remaining),full_edges=full,conservative=False,elastic=0.,reweighted=True)
        if proposal is not None:proposal=shrink(proposal,p)
        previous=value;clean=accept(proposal);trace.append(dict(phase='shape',route=route,public_domain=expanded,legal=clean is not None,**stats))
        if clean is not None:
            state=clean;overrides.clear()
        else:
            just_preconditioned=False;expanded=False
            if organized and not preconditioned and stats.get("status")=="PrimalInfeasible":
                t=time.perf_counter();state=obstacle_seed(p,state,0,preplaced_only=True,preserve_centers=True);fs=dict(seconds=time.perf_counter()-t);prefix=state.copy();overrides.clear();preconditioned=True;just_preconditioned=True
                trace.append(dict(phase="conditional_static_obstacle_projection",**fs))
            if organized and preconditioned and not just_preconditioned and stats.get("status")=="PrimalInfeasible":
                expanded=True
            if organized and not just_preconditioned and stats.get("conflicting_edges"):
                changed=0;center=state[:,:2]+state[:,2:]/2
                for a,b,axis,weight in stats["conflicting_edges"]:
                    key=(min(a,b),max(a,b))
                    if key in overrides or (p.preplaced[a] and p.preplaced[b]):continue
                    other=1-axis;first,second=(a,b) if center[a,other]<=center[b,other] else (b,a)
                    overrides[key]=(first,second,other);changed+=1
                    trace.append(dict(phase="dual_relation_repair",old=[a,b,axis],new=[first,second,other],dual_weight=weight))
                    if changed>=settings["maximum_edge_flips_per_failure"]:break
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
    return best,dict(seconds=time.perf_counter()-start,trace=trace,row_retained=bool(np.array_equal(best,fallback)),constructive_recovery=seeded,preconditioned=preconditioned,expanded=expanded,max_free_dimension_factor=float(np.maximum(best[~(p.fixed|p.preplaced),2:]/draft[~(p.fixed|p.preplaced),2:],draft[~(p.fixed|p.preplaced),2:]/best[~(p.fixed|p.preplaced),2:]).max()) if (~(p.fixed|p.preplaced)).any() else 1.,proxy=obj(best)),prefix
