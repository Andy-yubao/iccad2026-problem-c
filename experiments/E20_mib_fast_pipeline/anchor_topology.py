"""Repair globally inconsistent fixed-anchor paths in a public draft topology.

Every pair retains exactly one separation. Candidate switches respect the
original center ordering in their new axis, so both graphs remain acyclic.
The score uses only public locked coordinates and dimensions, never labels.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def repair(p,seed,edges,seconds=.06,max_switches=16):
    started=time.perf_counter();edges=np.array(edges,dtype=int).copy();n=p.n
    centers=seed[:,:2]+seed[:,2:]/2;fixed=np.flatnonzero(p.preplaced);trace=[]
    if not len(fixed):return edges,{'seconds':time.perf_counter()-started,'switches':trace,'violation':0.}
    def assess(relations,path=False):
        total=0.;worst=None
        for axis in (0,1):
            part=relations[relations[:,2]==axis];a,b=part[:,:2].T
            graph=csr_matrix((-seed[a,axis+2],(a,b)),shape=(n,n))
            distance,predecessor=shortest_path(graph,directed=True,method='FW',return_predecessors=True)
            starts=np.zeros(n);starts[fixed]=p.specs[fixed,axis]
            implied=starts[:,None]-distance[:,fixed]
            parent=np.argmax(implied,axis=0);lower=implied[parent,np.arange(len(fixed))]
            violation=np.maximum(0,lower-p.specs[fixed,axis]);total+=float(violation.sum())/p.scale
            if path and violation.max()>1e-7 and (worst is None or violation.max()>worst[0]):
                k=int(violation.argmax());source=int(parent[k]);target=int(fixed[k]);chain=[];at=target
                while at!=source:
                    previous=int(predecessor[source,at])
                    if previous<0:break
                    chain.append((previous,at,axis));at=previous
                worst=(float(violation[k]),chain)
        return total,[] if worst is None else worst[1]
    score,path=assess(edges,True);changed=set()
    for _ in range(max_switches):
        if score<1e-9 or not path or time.perf_counter()-started>=seconds:break
        best=None
        for a,b,axis in path:
            if p.preplaced[a] and p.preplaced[b]:continue
            index=max(a,b)*(max(a,b)-1)//2+min(a,b);other=1-axis
            if index in changed:continue
            order=(centers[a,other],a)<(centers[b,other],b)
            replacement=(a,b,other) if order else (b,a,other)
            candidate=edges.copy();candidate[index]=replacement;new_score,_=assess(candidate)
            penetration=max(0.,(seed[replacement[0],other]+seed[replacement[0],other+2]-seed[replacement[1],other]))/p.scale
            rank=(round(new_score,9),penetration)
            if new_score<=score+1e-9 and (best is None or rank<best[4]):best=(new_score,candidate,replacement,(a,b,axis),rank,index)
            if new_score<1e-9 or time.perf_counter()-started>=seconds:break
        if best is None:break
        score,edges,new,old,_,index=best;changed.add(index);trace.append({'old':old,'new':new,'violation':score})
        score,path=assess(edges,True)
    return edges,{'seconds':time.perf_counter()-started,'switches':trace,'violation':score}
