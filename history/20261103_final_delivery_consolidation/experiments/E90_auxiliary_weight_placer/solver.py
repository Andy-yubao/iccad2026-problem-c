"""Budgeted vectorized geometry controls, immutable model and legal incumbent."""
import time
import numpy as np
import math
from .backend.frozen.packing import prepare_problem,_dimensions
from .backend.frozen.feasible import Objective,row_control
from .backend.frozen.solver import hard_valid
from .backend.guard import repair_tiny_negative
from .fast_geometry import topology,step,contact_tree,reduce_edges
from .canonical import canonicalize
from .projection import solve as project


def raw_measures(x,p):
    center=x[:,:2]+x[:,2:]/2;wire=0.
    if len(p.b2b):
        i,j=p.b2b[:,:2].astype(int).T;wire+=float((np.abs(center[i]-center[j]).sum(1)*p.b2b[:,2]).sum())
    if len(p.p2b):
        i,j=p.p2b[:,:2].astype(int).T;wire+=float((np.abs(p.pins[i]-center[j]).sum(1)*p.p2b[:,2]).sum())
    area=float(np.prod((x[:,:2]+x[:,2:]).max(0)-x[:,:2].min(0)))
    return area,wire


def optimize(draft,inputs,*,iterations=4,mode=0,projection=True,seconds=.21,
             boundary=12.,contact=6.,reweighted=True,relaxed=False,hard=False,reserve=0.,snap=1e-6,
             group_order=False,elastic_strength=100000.,adaptive=False,draft_proxy=False,rescue=False,
             repair_edges=False,hybrid=False,near_seed=False,seed_order=0,safe_mode=False,plateau=False,
             preserve_centers=False,anchor_route=False,shape_square=False,fast_budget=False,late_extension=False,
             robust_seeds=False,periodic_project=False,hard_tangent=False,certified=False,quick_project=False,wire_exact=False,
             quadratic_project=False,incumbent_project=False,warm_project=False,early_public=False,selection_draft=False):
    start=time.perf_counter();p=prepare_problem(*inputs);native_objective=Objective(p);obj=native_objective
    if adaptive:
        iterations,seconds,reserve=(64,.09,.025) if p.n<60 else ((64,.17,.045) if p.n<90 else ((64,.34,.065) if p.n<110 else (64,.55,.08)))
        if fast_budget:
            seconds,reserve=(.055,.015) if p.n<60 else ((.105,.025) if p.n<90 else ((.20,.04) if p.n<110 else (.36,.06)))
    anchors=None
    if draft_proxy or selection_draft:
        anchor_area,anchor_wire=raw_measures(draft,p)
        if draft_proxy:anchors=(max(anchor_area/p.scale**2,1e-8),max(anchor_wire/p.scale,1e-8))
        objective=obj
        def normalized(x):
            a,w=raw_measures(x,p)
            return math.log(1+.5*max(0,a/anchor_area-1)+.5*max(0,w/max(anchor_wire,1e-8)-1))+2*sum(objective.violations(x))/objective.nsoft
        obj=normalized
    fallback=row_control(p);best=fallback.copy();value=obj(best);trace=[];overrides={};infeasible=False;legal_steps=0
    widths,heights=_dimensions(p,np.ones(p.n) if shape_square else draft[:,2]/draft[:,3],True)
    state=np.column_stack((draft[:,:2],widths,heights))
    if preserve_centers:
        state[~p.preplaced,:2]=np.maximum(0,draft[~p.preplaced,:2]+draft[~p.preplaced,2:]/2-state[~p.preplaced,2:]/2)
    if near_seed or anchor_route:
        from .legal_seed import seed
        state=seed(p,state if shape_square else draft,seed_order,preplaced_only=anchor_route,preserve_centers=preserve_centers)
    def accept(x,label):
        nonlocal best,value
        if x is None:return False
        x=canonicalize(x,p,tolerance=snap);x,_=repair_tiny_negative(x,p)
        if not hard_valid(x,p):return False
        score=obj(x)
        if score<value:
            best=x.copy();value=score
        return x
    accept(draft,'raw')
    if near_seed:accept(state,'near_seed')
    def relations(state):
        full=topology(state,p.constraints,mode=mode,array=True,group_order=group_order)
        if safe_mode:
            a,b,c=full.T
            invalid=state[b,c]-state[a,c]-state[a,c+2]<-1e-7
            physical=topology(state,p.constraints,mode=0,array=True)
            full[invalid]=physical[invalid]
        for (i,j),edge in overrides.items():full[max(i,j)*(max(i,j)-1)//2+min(i,j)]=edge
        return full
    if warm_project:
        full=relations(state);tree=contact_tree(state,p,full)
        proposed,stats=project(p,state,reduce_edges(full,p.n),tree,
            seconds=min(.05,seconds/3),boundary=boundary,contact=contact,anchors=anchors,quadratic=True)
        clean=accept(proposed,'warm_project');trace.append({'phase':'warm_project','legal':isinstance(clean,np.ndarray),**stats})
        if isinstance(clean,np.ndarray):state=clean
    stale=0;extended=False;gate_attempted=False;public_stopped=False
    draft_wire=raw_measures(draft,p)[1] if early_public else None
    for iteration in range(iterations):
        left=seconds-(time.perf_counter()-start)
        if left<.015+reserve:
            if quick_project and not gate_attempted and np.array_equal(best,fallback):
                gate_attempted=True;pf=relations(state)
                px,ps=project(p,state,reduce_edges(pf,p.n),[],seconds=.06 if p.n<90 else .13,boundary=8.,contact=0.,anchors=anchors,quadratic=quadratic_project)
                pc=accept(px,'early_projection')
                trace.append({'phase':'early_projection','legal':isinstance(pc,np.ndarray),**ps})
                if isinstance(pc,np.ndarray):state=pc;break
            if late_extension and not extended and np.array_equal(best,fallback):
                seconds=max(seconds,1.4 if p.n>=90 else .7);reserve=.1 if p.n>=90 else .05;extended=True
                left=seconds-(time.perf_counter()-start)
            else:break
        full=relations(state);edges=reduce_edges(full,p.n)
        proposal,stats=step(p,state,edges,seconds=max(.01,left-reserve),boundary=boundary,contact=contact,
            reweighted=reweighted,conservative=(not relaxed or (hybrid and legal_steps>=2)),full_edges=full,elastic=0. if hard else elastic_strength*(1+iteration/3),anchors=anchors,wire_exact=wire_exact)
        previous_value=value;clean=accept(proposal,'shape')
        trace.append({'phase':'shape','legal':isinstance(clean,np.ndarray),**stats})
        if proposal is None:
            infeasible=stats['status']=='PrimalInfeasible'
            if repair_edges and stats.get('conflicting_edges'):
                candidates=stats['conflicting_edges'];changed=0
                for a,b,axis,weight in candidates:
                    key=(min(a,b),max(a,b))
                    if key in overrides:continue
                    other=1-axis;centers=state[:,:2]+state[:,2:]/2
                    first,second=(a,b) if centers[a,other]<=centers[b,other] else (b,a)
                    overrides[key]=(first,second,other);changed+=1
                    trace.append({'phase':'conflict_repair','old':[a,b,axis],'new':[first,second,other],'dual_weight':weight})
                    if changed==2:break
                if changed:continue
            break
        infeasible=False
        legal_steps+=int(isinstance(clean,np.ndarray))
        state=clean if isinstance(clean,np.ndarray) else proposal
        if early_public and isinstance(clean,np.ndarray) and sum(native_objective.violations(best))==0:
            area,wire=raw_measures(best,p)
            if area<=1.025*p.total_area and wire<=max(draft_wire,1e-12):
                trace.append({'phase':'public_stop','area_over_total':area/p.total_area,'wire_over_draft':wire/max(draft_wire,1e-12)})
                public_stopped=True;break
        if periodic_project and iteration%4==3 and seconds-(time.perf_counter()-start)>.05:
            pf=relations(state);pt=contact_tree(state,p,pf)
            px,ps=project(p,state,reduce_edges(pf,p.n),pt,seconds=min(.15,seconds-(time.perf_counter()-start)-.02),
                boundary=boundary,contact=contact,anchors=anchors,hard_tangent=hard_tangent,quadratic=quadratic_project)
            pc=accept(px,'periodic_project');trace.append({'phase':'periodic_project','legal':isinstance(pc,np.ndarray),**ps})
            if isinstance(pc,np.ndarray):state=pc
        if plateau and isinstance(clean,np.ndarray):
            stale=stale+1 if previous_value-value<1e-5 else 0
            if stale>=2 and legal_steps>=4 and sum(native_objective.violations(best))==0:break
    if projection and not public_stopped and not infeasible and seconds-(time.perf_counter()-start)>.015:
        if incumbent_project and not np.array_equal(best,fallback):state=best.copy()
        full=relations(state);tree=contact_tree(state,p,full)
        proposed,stats=project(p,state,reduce_edges(full,p.n),tree,contact=contact,boundary=boundary,
            seconds=max(.01,seconds-(time.perf_counter()-start)),anchors=anchors,hard_tangent=hard_tangent,quadratic=quadratic_project)
        legal=accept(proposed,'project');trace.append({'phase':'project','legal':isinstance(legal,np.ndarray),**stats})
    if robust_seeds and (np.array_equal(best,fallback) or native_objective(best)>.9):
        from .backend.rescue import seed_pack
        for seed_mode in [2,0,1]:
            packed=seed_pack(draft,inputs,seed_mode,seconds=.8)
            if packed is None:continue
            accept(packed,'full_packed_seed')
            refined,rs=optimize(packed,inputs,iterations=48,seconds=.8,reserve=.1,mode=1,
                reweighted=False,boundary=40.,contact=20.,snap=1e-4,periodic_project=True,hard_tangent=True)
            accept(refined,'full_packed_refine');trace.append({'phase':'full_packed_refine','mode':seed_mode,'seconds':rs['seconds']})
            if native_objective(best)<.6:break
    if rescue and np.array_equal(best,fallback):
        alternate,alternate_stats=optimize(draft,inputs,iterations=6,mode=1-mode,projection=True,
            seconds=.23,hard=True,relaxed=True,reserve=.05,snap=snap,repair_edges=True)
        accept(alternate,'alternate_relations');trace.append({'phase':'alternate_relations','seconds':alternate_stats['seconds']})
    if rescue and np.array_equal(best,fallback):
        from .backend.rescue import seed_pack
        for seed_mode in [2,0]:
            packed=seed_pack(draft,inputs,seed_mode,seconds=.08)
            if packed is None:continue
            accept(packed,'packed_rescue')
            refined,substats=optimize(packed,inputs,iterations=4,mode=1,projection=True,
                seconds=.14,hard=True,relaxed=True,reserve=.04,snap=snap,repair_edges=True)
            accept(refined,'packed_polish');trace.append({'phase':'packed_rescue','mode':seed_mode,'seconds':substats['seconds']})
            if obj(best)<.5:break
    if certified:
        from .certify_polish import polish
        best,cs=polish(best,p);trace.append({'phase':'certified_rigid_moves',**cs});value=obj(best)
    assert hard_valid(best,p)
    return best,{'seconds':time.perf_counter()-start,'proxy':value,'trace':trace,
        'row_retained':bool(np.array_equal(best,fallback)),'public_stopped':public_stopped}
