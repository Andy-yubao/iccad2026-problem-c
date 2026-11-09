"""One explicit state machine for full ordering, obstacle projection, and cuts."""
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
from ..R4.normalization import shrink
from ..R13.cuts import update as update_cuts


def optimize(draft,inputs,settings,organized=False):
    start=time.perf_counter();deadline=start+settings['seconds']
    p=prepare_problem(*inputs);obj=Objective(p);fallback=row_control(p)
    best=fallback.copy();value=obj(best);trace=[];useful=False
    state=np.array(draft,copy=True);w,h=_dimensions(p,state[:,2]/state[:,3],True)
    state[:,2:]=np.column_stack((w,h));state[p.preplaced]=p.specs[p.preplaced];prefix=state.copy()
    pending=None;preconditioned=False;seeded=False;cut_mode=False
    cuts={};removed=set();cut_initialized=False;overrides={}

    def accept(candidate):
        nonlocal best,value,useful
        if candidate is None:return None
        candidate=canonicalize(candidate,p,tolerance=1e-4);candidate,_=repair_tiny_negative(candidate,p)
        if not hard_valid(candidate,p):return None
        score=obj(candidate)
        if score<value:best=candidate.copy();value=score;useful=True
        return candidate

    accept(draft);accept(state)
    for iteration in range(settings['recovery_maximum_shape_solves']):
        if iteration>=settings['maximum_shape_solves'] and not cut_mode:break
        remaining=deadline-time.perf_counter()-settings['projection_and_polish_reserve_seconds']
        if remaining<.018:break
        full=topology(state,p.constraints,mode=0,array=True)
        for (i,j),edge in overrides.items():full[max(i,j)*(max(i,j)-1)//2+min(i,j)]=edge
        was_cut=cut_mode
        active=update_cuts(state,p,full,cuts,removed,not cut_initialized) if cut_mode else full
        if cut_mode:cut_initialized=True
        proposal,stats=step(p,state,reduce_edges(active,p.n),seconds=min(settings['shape_cap_seconds'],remaining),full_edges=full,conservative=False,elastic=0.,reweighted=True)
        if proposal is not None:
            proposal=shrink(proposal,p);removed.clear()
        previous=value;clean=accept(proposal)
        trace.append(dict(phase='shape',sparse=was_cut,active_pair_cuts=len(active),legal=clean is not None,**stats))
        if clean is not None:
            state=clean;overrides.clear()
            if useful:pending=None
        else:
            if not useful and proposal is not None:
                pending=proposal.copy();state=proposal.copy()
                trace.append(dict(phase='pending_shape'))
            status=stats.get('status');just_projected=False
            if not preconditioned and status=='PrimalInfeasible':
                t=time.perf_counter();state=obstacle_seed(p,state,0,preplaced_only=True,preserve_centers=True)
                prefix=state.copy();overrides.clear();preconditioned=True;just_projected=True
                trace.append(dict(phase='conditional_static_obstacle_projection',seconds=time.perf_counter()-t))
            elif preconditioned and not cut_mode and status=='PrimalInfeasible':
                cut_mode=True;overrides.clear()
                trace.append(dict(phase='start_collision_cuts'))
            if was_cut and stats.get('conflicting_edges'):
                for a,b,axis,weight in stats['conflicting_edges']:
                    key=(min(a,b),max(a,b));cuts.pop(key,None);removed.add(key)
                    trace.append(dict(phase='drop_conflict_cut',pair=list(key),dual_weight=weight))
            elif not cut_mode and not just_projected and stats.get('conflicting_edges'):
                center=state[:,:2]+state[:,2:]/2;changed=0
                for a,b,axis,weight in stats['conflicting_edges']:
                    key=(min(a,b),max(a,b))
                    if key in overrides or (p.preplaced[a] and p.preplaced[b]):continue
                    other=1-axis;first,second=(a,b) if center[a,other]<=center[b,other] else (b,a)
                    overrides[key]=(first,second,other);changed+=1
                    trace.append(dict(phase='dual_relation_repair',old=[a,b,axis],new=[first,second,other],dual_weight=weight))
                    if changed>=settings['maximum_edge_flips_per_failure']:break
            if useful:state=best.copy()
        if iteration>=1 and useful and previous-value<1e-5 and sum(obj.violations(best))==0:break

    if not useful and pending is not None:
        left=deadline-time.perf_counter()-.010
        if left>.005:
            full=topology(pending,p.constraints,mode=0,array=True)
            proposed,ps=project(p,pending,reduce_edges(full,p.n),contact_tree(pending,p,full),seconds=min(settings['pending_projection_cap_seconds'],left),quadratic=True)
            clean=accept(proposed);trace.append(dict(phase='pending_projection',legal=clean is not None,**ps))
        else:trace.append(dict(phase='pending_projection_skipped',reason='deadline_reserve'))
    if not useful:
        t=time.perf_counter();clean=accept(seed(p,pending if pending is not None else prefix,0,preserve_centers=True));seeded=True
        trace.append(dict(phase='terminal_legal_seed',seconds=time.perf_counter()-t,legal=clean is not None))
    left=deadline-time.perf_counter()-.008
    if left>.008:
        full=topology(best,p.constraints,mode=0,array=True)
        proposed,ps=project(p,best,reduce_edges(full,p.n),contact_tree(best,p,full),seconds=left,quadratic=True)
        clean=accept(proposed);trace.append(dict(phase='project',legal=clean is not None,**ps))
    left=deadline-time.perf_counter()
    if left>.001:
        best,ps=polish(best,p,seconds=min(.008,left),max_accept=4,push=True)
        trace.append(dict(phase='certified_push',**ps))
    assert hard_valid(best,p)
    return best,dict(seconds=time.perf_counter()-start,trace=trace,row_retained=bool(np.array_equal(best,fallback)),constructive_recovery=seeded,preconditioned=preconditioned,cut_recovery=cut_mode,proxy=obj(best)),prefix
