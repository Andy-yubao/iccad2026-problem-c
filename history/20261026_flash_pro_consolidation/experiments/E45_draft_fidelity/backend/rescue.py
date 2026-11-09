"""Public-input rescue hypotheses, with a legal monotone incumbent."""
import time
import numpy as np
from .frozen.packing import prepare_problem, pack, _dimensions, default_order
from .frozen.feasible import Objective
from .frozen.global_refine import topology, solve_lp, contact_tree
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .conic_geometry import step
from .reduction import reduce_edges
from .guard import repair_tiny_negative


def measures(x, p):
    center = x[:, :2] + x[:, 2:] / 2
    wire = 0.
    if len(p.b2b):
        i,j=p.b2b[:,:2].astype(int).T
        wire += float((np.abs(center[i]-center[j]).sum(1)*p.b2b[:,2]).sum())
    if len(p.p2b):
        i,j=p.p2b[:,:2].astype(int).T
        wire += float((np.abs(p.pins[i]-center[j]).sum(1)*p.p2b[:,2]).sum())
    area = float(np.prod((x[:,:2]+x[:,2:]).max(0)-x[:,:2].min(0)))
    return np.array([area, wire, *Objective(p).violations(x)],dtype=float)


def clean(x,p):
    if x is None:
        return None
    x=canonicalize(x,p)
    x,_=repair_tiny_negative(x,p)
    return x if hard_valid(x,p) else None


def refine(seed, inputs, *, seconds=5., mode=1, boundary=12., contact=6., hard=False):
    begin=time.perf_counter();p=prepare_problem(*inputs);obj=Objective(p)
    state=np.asarray(seed).copy();best=clean(state,p);value=obj(best) if best is not None else float('inf')
    trace=[]
    for iteration in range(32):
        left=seconds-(time.perf_counter()-begin)-.025
        if left<.09:
            break
        edges=reduce_edges(topology(state,p.constraints,mode=mode),p.n)
        proposal,stats=step(p,state,edges,seconds=left,boundary=boundary,contact=contact,
                            elastic=0. if hard else 100000*(1+iteration/3),
                            reweighted=False,conservative=True,quadratic_wire=True,diagonal_wire=True)
        candidate=clean(proposal,p)
        if candidate is not None and obj(candidate)<value:
            best=candidate.copy();value=obj(best)
        trace.append({'iteration':iteration,'phase':'shape','legal':candidate is not None,
                      'proxy':obj(candidate) if candidate is not None else None,**stats})
        if proposal is None:
            break
        state=proposal
        if iteration%4==3:
            left=seconds-(time.perf_counter()-begin)-.025
            if left<.06:
                break
            full=topology(state,p.constraints,mode=mode)
            tree=contact_tree(state,p,full)
            projection,stats=solve_lp(p,state,reduce_edges(full,p.n),boundary=boundary,
                                      contact=contact,contacts=tree,limit=min(.5,left))
            candidate=clean(projection,p)
            if candidate is not None:
                state=candidate
                if obj(candidate)<value:
                    best=candidate.copy();value=obj(best)
            trace.append({'iteration':iteration,'phase':'project','legal':candidate is not None,**stats})
    return best,{'seconds':time.perf_counter()-begin,'trace':trace,'proxy':value,'mode':mode,
                 'boundary':boundary,'contact':contact,'hard':hard}


def seed_pack(draft,inputs,mode=0,seconds=1.):
    p=prepare_problem(*inputs)
    centers=draft[:,:2]+draft[:,2:]/2
    ratios=draft[:,2]/draft[:,3] if mode<2 else np.ones(p.n)
    if mode==0:
        order=sorted(default_order(p).tolist(),key=lambda i:(round(centers[i,1]/max(p.scale*.15,1e-9)),centers[i,0],i))
    elif mode==1:
        order=default_order(p)
    else:
        order=sorted(default_order(p).tolist(),key=lambda i:(round(centers[i,0]/max(p.scale*.15,1e-9)),centers[i,1],i))
    x=pack(p,order=order,ratios=ratios,target_width=p.scale,shape_trials=1,
           proposal_centers=centers if mode!=1 else None,deadline=time.monotonic()+seconds)
    return clean(x,p)
