"""Retain alternative direction repairs when one-step greedy repair stalls."""
import heapq
import itertools
import time
import numpy as np
from .frozen.packing import prepare_problem,_dimensions
from .frozen.feasible import Objective,row_control
from .frozen.global_refine import topology,solve_lp,contact_tree
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .certificate import infeasibility_certificate
from .topology_repair import graph_layout
from .reduction import reduce_edges
from .guard import repair_tiny_negative


def optimize(raw,inputs,seconds=1.3):
    begin=time.perf_counter();deadline=begin+seconds
    p=prepare_problem(*inputs);objective=Objective(p)
    seed=np.column_stack((np.asarray(raw)[:,:2],*_dimensions(p,np.asarray(raw)[:,2]/np.asarray(raw)[:,3],True)))
    centers=seed[:,:2]+seed[:,2:]/2
    best=row_control(p);value=objective(best)
    if not hard_valid(best,p):raise RuntimeError('Invalid public fallback')
    events=[{'elapsed':time.perf_counter()-begin,'proxy':value,'positions':best.tolist()}]
    trace=[];queue=[];serial=itertools.count();seen=set();expanded=0;feasible_graphs=0
    def accept(pos,phase):
        nonlocal best,value
        if pos is None:return False
        pos=canonicalize(pos,p);pos,_=repair_tiny_negative(pos,p)
        if not hard_valid(pos,p):return False
        score=objective(pos)
        if score<value-1e-10:
            best,value=pos.copy(),score
            events.append({'elapsed':time.perf_counter()-begin,'proxy':float(value),'phase':phase,'positions':best.tolist()})
        return True
    for mode in (0,1):
        edges=topology(seed,p.constraints,mode)
        signature=tuple(edges)
        if signature in seen:continue
        seen.add(signature)
        layout,violation=graph_layout(p,seed,edges)
        if not np.isfinite(violation):continue
        heapq.heappush(queue,(violation,0,0,next(serial),edges,layout))
    while queue and expanded<48 and deadline-time.perf_counter()>.07:
        violation,depth,changes,_,edges,layout=heapq.heappop(queue);expanded+=1
        if violation<1e-9:
            feasible_graphs+=1;accept(layout,'beam_construct')
            reduced=reduce_edges(edges,p.n);tree=contact_tree(layout,p,edges)
            for contact in (8.,0.):
                remaining=deadline-time.perf_counter()-.025
                if remaining<.04:break
                candidate,stats=solve_lp(p,layout,reduced,contact=contact,contacts=tree if contact else (),limit=min(.3,remaining))
                legal=accept(candidate,'beam_fixed_lp')
                trace.append({'phase':'fixed_lp','contact':contact,'legal':legal,'edges':len(reduced),**stats})
            if feasible_graphs>=4:break
            continue
        certificate=infeasibility_certificate(p,seed,edges)
        if not certificate or 'path' not in certificate:continue
        path=certificate['path'];axis=certificate['axis']
        positions={frozenset((i,j)):k for k,(i,j,a) in enumerate(edges)}
        proposals=[]
        for i,j in zip(path,path[1:]):
            index=positions[frozenset((i,j))];other=1-axis
            preferred=(i,j,other) if centers[i,other]<=centers[j,other] else (j,i,other)
            for replacement in (preferred,(preferred[1],preferred[0],other)):
                if deadline-time.perf_counter()<.06:break
                proposal=edges.copy();proposal[index]=replacement;signature=tuple(proposal)
                if signature in seen:continue
                seen.add(signature)
                candidate,score=graph_layout(p,seed,proposal)
                if not np.isfinite(score):continue
                # Unlike the former greedy rule, keep neutral or temporarily worse branches.
                proposals.append((score,depth+1,changes+1,next(serial),proposal,candidate))
        proposals.sort(key=lambda item:item[:4])
        for item in proposals[:6]:heapq.heappush(queue,item)
        if len(queue)>32:queue=heapq.nsmallest(32,queue);heapq.heapify(queue)
    trace.append({'phase':'beam_summary','expanded':expanded,'seen':len(seen),'feasible_graphs':feasible_graphs,
                  'elapsed':time.perf_counter()-begin})
    if not hard_valid(best,p):raise AssertionError('Invalid strict incumbent')
    return best,{'seconds':time.perf_counter()-begin,'proxy':float(value),'trace':trace,'guard_fallback':False,'improvements':events}
