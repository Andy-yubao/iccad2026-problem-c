"""Route B: public-anchor certificates guide discrete separation repair."""
from collections import deque
import time
import numpy as np
from .frozen.packing import prepare_problem, _dimensions
from .frozen.global_refine import topology, solve_lp, contact_tree
from .frozen.feasible import Objective, row_control
from .frozen.solver import hard_valid
from .frozen.canonical_geometry import canonicalize
from .certificate import infeasibility_certificate
from .reduction import reduce_edges


def graph_layout(p,seed,edges):
    """Longest-path placement, plus total public-anchor contradiction magnitude."""
    out=seed.copy();score=0.
    for axis in (0,1):
        neighbors=[[] for _ in range(p.n)];degree=[0]*p.n
        for i,j,a in edges:
            if a==axis:neighbors[i].append(j);degree[j]+=1
        ready=deque(i for i in range(p.n) if degree[i]==0)
        lower=np.zeros(p.n);lower[p.preplaced]=p.specs[p.preplaced,axis]
        count=0
        while ready:
            i=ready.popleft();count+=1
            for j in neighbors[i]:
                lower[j]=max(lower[j],lower[i]+seed[i,axis+2])
                degree[j]-=1
                if degree[j]==0:ready.append(j)
        if count!=p.n:return None,float('inf')
        score+=float(np.maximum(lower[p.preplaced]-p.specs[p.preplaced,axis],0.).sum())/p.scale
        out[:,axis]=lower
    return out,score


def optimize(raw,inputs,seconds=1.5):
    begin=time.perf_counter();deadline=begin+seconds
    p=prepare_problem(*inputs);objective=Objective(p)
    best=row_control(p);best_value=objective(best);trace=[]
    if not hard_valid(best,p):raise RuntimeError('Invalid public row fallback')
    events=[{'elapsed':time.perf_counter()-begin,'proxy':best_value,'positions':best.tolist()}]
    def accept(pos,phase):
        nonlocal best,best_value
        if pos is None:return False
        pos=canonicalize(pos,p)
        xy=pos[:,:2];xy[(xy<0)&(xy>=-1e-9)]=0.
        if not hard_valid(pos,p):return False
        value=objective(pos)
        if value<best_value-1e-10:
            best,best_value=pos.copy(),value
            events.append({'elapsed':time.perf_counter()-begin,'phase':phase,'proxy':float(value),'positions':best.tolist()})
        return True
    seed=np.column_stack((np.asarray(raw)[:,:2],*_dimensions(p,np.asarray(raw)[:,2]/np.asarray(raw)[:,3],True)))
    center=seed[:,:2]+seed[:,2:]/2
    for mode in (0,1):
        if deadline-time.perf_counter()<.08:break
        edges=topology(seed,p.constraints,mode)
        layout,score=graph_layout(p,seed,edges)
        repairs=0
        for iteration in range(24):
            if score<1e-9 or deadline-time.perf_counter()<.12:break
            certificate=infeasibility_certificate(p,seed,edges)
            if not certificate or 'path' not in certificate:break
            axis=certificate['axis'];path=certificate['path']
            lookup={(i,j,a):k for k,(i,j,a) in enumerate(edges)}
            winner=None
            for i,j in zip(path,path[1:]):
                old_index=lookup[(i,j,axis)]
                other=1-axis
                preferred=(i,j,other) if center[i,other]<=center[j,other] else (j,i,other)
                for replacement in (preferred,(preferred[1],preferred[0],other)):
                    if deadline-time.perf_counter()<.10:break
                    proposal=edges.copy();proposal[old_index]=replacement
                    candidate,current=graph_layout(p,seed,proposal)
                    if current < score-1e-10 and (winner is None or current<winner[0]):
                        winner=(current,proposal,candidate)
            if winner is None:break
            score,edges,layout=winner;repairs+=1
        legal=score<1e-9 and accept(layout,'repaired_longest_path')
        trace.append({'phase':'discrete_repair','mode':mode,'repairs':repairs,'remaining_anchor_violation':score if np.isfinite(score) else None,
                      'legal':legal,'elapsed':time.perf_counter()-begin})
        if not legal:continue
        reduced=reduce_edges(edges,p.n)
        tree=contact_tree(layout,p,edges)
        for contact in (8.,0.):
            remaining=deadline-time.perf_counter()-.025
            if remaining<.03:break
            before=time.perf_counter()
            candidate,stats=solve_lp(p,layout,reduced,contact=contact,contacts=tree if contact else (),limit=min(.35,remaining))
            accept(candidate,'repaired_position_lp')
            trace.append({'phase':'repaired_position_lp','seconds':time.perf_counter()-before,'edges':len(reduced),**stats})
    if not hard_valid(best,p):raise AssertionError('Invalid strict incumbent')
    return best,{'seconds':time.perf_counter()-begin,'proxy':float(best_value),'trace':trace,'guard_fallback':False,'improvements':events}
