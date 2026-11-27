"""Shape-first route: spend short budget on joint correction, then projection."""
import time
import numpy as np
from .frozen.packing import prepare_problem, _dimensions
from .frozen.feasible import Objective, row_control
from .frozen.global_refine import topology, solve_lp
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .observe import invalid_reasons
from .reduction import reduce_edges
from .sparse_shape import step


def optimize(raw, inputs, seconds=1.5, linear_area=False):
    begin=time.perf_counter();deadline=begin+seconds
    p=prepare_problem(*inputs);objective=Objective(p)
    best=row_control(p)
    if not hard_valid(best,p):raise RuntimeError('Invalid public row fallback')
    value=objective(best);trace=[]
    events=[{'elapsed':time.perf_counter()-begin,'proxy':value,'positions':best.tolist()}]
    def accept(candidate,label):
        nonlocal best,value
        if candidate is None:return False
        candidate=canonicalize(candidate,p)
        xy=candidate[:,:2];xy[(xy<0)&(xy>=-1e-9)]=0.
        if not hard_valid(candidate,p):return False
        score=objective(candidate)
        if score<value-1e-10:
            best=candidate.copy();value=score
            events.append({'elapsed':time.perf_counter()-begin,'proxy':float(value),'phase':label,'positions':best.tolist()})
        return True
    if hard_valid(raw,p):accept(np.asarray(raw).copy(),'raw')
    dims=_dimensions(p,np.asarray(raw)[:,2]/np.asarray(raw)[:,3],True)
    state=np.column_stack((np.asarray(raw)[:,:2],*dims))
    # Deliberately skip the draft's fixed-shape LPs and partial pack.
    # A usable joint step and a shape-preserving projection get priority.
    for iteration in range(5):
        remaining=deadline-time.perf_counter()-.025
        if remaining<.12:break
        before=time.perf_counter()
        full=topology(state,p.constraints);edges=reduce_edges(full,p.n)
        reserved=min(.28,max(.08,remaining*.22))
        if linear_area:
            from .linear_shape import step as propose
        else:
            propose=step
        candidate,stats=propose(p,state,edges,seconds=max(.03,remaining-reserved),
                             reweighted=True,elastic=100000*(1+iteration/3),iteration=iteration,tangents=13)
        legal=accept(candidate,'joint_shape')
        trace.append({'phase':'joint_shape','iteration':iteration,'seconds':time.perf_counter()-before,
                      'edges':len(edges),'full_edges':len(full),'legal':legal,**stats,
                      'rejection':invalid_reasons(candidate,p) if candidate is not None and not legal else None})
        if candidate is None:break
        state=candidate
        remaining=deadline-time.perf_counter()-.025
        if remaining<.025:break
        before=time.perf_counter()
        full=topology(candidate,p.constraints);edges=reduce_edges(full,p.n)
        projected,stats=solve_lp(p,candidate,edges,contact=0.,limit=min(.28,remaining))
        legal=accept(projected,'shape_projection')
        trace.append({'phase':'shape_projection','iteration':iteration,'seconds':time.perf_counter()-before,
                      'edges':len(edges),'legal':legal,**stats})
        if legal:state=projected
    if not hard_valid(best,p):raise AssertionError('Invalid strict incumbent')
    return best,{'seconds':time.perf_counter()-begin,'proxy':float(value),'trace':trace,
                 'guard_fallback':False,'improvements':events}
