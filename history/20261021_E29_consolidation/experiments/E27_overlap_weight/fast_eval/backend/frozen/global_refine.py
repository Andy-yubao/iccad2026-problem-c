"""Public-input global topology/geometry postprocessing. No reference arguments.

Fixed dimensions give an LP: all-pair separation, dynamic bbox, boundary slacks,
weighted Manhattan wires, geometric-contact candidates and optional displacement.
Only local experiment algorithms are imported.
"""
from __future__ import annotations
import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from .packing import prepare_problem,_dimensions
from .feasible import Objective,is_legal
from .postprocess import finalize


def topology(pos,c,mode=0):
    """Select directions from draft; boundary-sharing pairs choose tangential axis."""
    center=pos[:,:2]+pos[:,2:]/2;n=len(pos);edges=[]
    for i in range(n):
        for j in range(i):
            delta=center[i]-center[j];extent=(pos[i,2:]+pos[j,2:])/2
            margin=np.abs(delta)-extent
            axis=int(np.argmax(margin/np.maximum(extent,1e-12)))
            a,b=int(c[i,4]),int(c[j,4])
            if mode:
                if (a&b&3):axis=1
                elif (a&b&12):axis=0
            sign=delta[axis]>=0
            if mode:
                bits=(1,2) if axis==0 else (8,4)
                if a&bits[0] or b&bits[1]:sign=False
                elif b&bits[0] or a&bits[1]:sign=True
            before,after=(j,i) if sign else (i,j)
            edges.append((before,after,axis))
    return edges


def solve_lp(p,seed,edges,*,contact=0.,boundary=8.,movement=.01,wire=1.,area=1.,limit=3.,contacts=()):
    scale=p.scale;n=p.n;z=seed/scale;wh=z[:,2:];c=p.constraints
    # x,y per block; bbox loX,loY,hiX,hiY; boundary slack per block; wire epigraph.
    obj=[];bounds=[];rows=[];rhs=[]
    def var(cost=0.,lo=0.,hi=None):
        k=len(obj);obj.append(cost);bounds.append((lo,hi));return k
    xy=np.array([[var(),var()] for _ in range(n)])
    lo=np.array([var(-area),var(-area)]);hi=np.array([var(area),var(area)])
    def le(terms,value):rows.append(terms);rhs.append(value)
    def eq(terms,value):le(terms,value);le({k:-v for k,v in terms.items()},-value)
    for i in range(n):
        for a in (0,1):
            le({lo[a]:1,xy[i,a]:-1},0)
            le({xy[i,a]:1,hi[a]:-1},-wh[i,a])
            if p.preplaced[i]:bounds[xy[i,a]]=(z[i,a],z[i,a])
            if movement and not p.preplaced[i]:
                d=var(movement/n);le({xy[i,a]:1,d:-1},z[i,a]);le({xy[i,a]:-1,d:-1},-z[i,a])
        if c[i,4]:
            slack=var(boundary/max(Objective(p).nsoft,1))
            for bit,a,upper in [(1,0,False),(2,0,True),(4,1,True),(8,1,False)]:
                if not c[i,4]&bit:continue
                if upper:le({hi[a]:1,xy[i,a]:-1,slack:-1},wh[i,a])
                else:le({xy[i,a]:1,lo[a]:-1,slack:-1},0)
    for i,j,a in edges:le({xy[i,a]:1,xy[j,a]:-1},-wh[i,a])
    norm=max(p.total_wire_weight,1e-12)
    for i,j,w in p.b2b:
        i,j=int(i),int(j)
        for a in (0,1):
            d=var(wire*w/norm);offset=(wh[i,a]-wh[j,a])/2
            le({xy[i,a]:1,xy[j,a]:-1,d:-1},-offset)
            le({xy[i,a]:-1,xy[j,a]:1,d:-1},offset)
    for pin,i,w in p.p2b:
        i,pin=int(i),int(pin)
        for a in (0,1):
            d=var(wire*w/norm);target=p.pins[pin,a]/scale-wh[i,a]/2
            le({xy[i,a]:1,d:-1},target);le({xy[i,a]:-1,d:-1},-target)
    for i,j,a in contacts:
        gap=var(contact/max(Objective(p).nsoft,1));t=1-a
        le({xy[j,a]:1,xy[i,a]:-1,gap:-1},wh[i,a])
        # Positive tangential intersection; contact gap remains a soft objective.
        overlap=.02*min(wh[i,t],wh[j,t])
        le({xy[i,t]:1,xy[j,t]:-1},wh[j,t]-overlap)
        le({xy[j,t]:1,xy[i,t]:-1},wh[i,t]-overlap)
    ii=[];jj=[];data=[]
    for r,row in enumerate(rows):
        for k,v in row.items():ii.append(r);jj.append(k);data.append(v)
    matrix=coo_matrix((data,(ii,jj)),shape=(len(rows),len(obj))).tocsr()
    result=linprog(obj,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs',
        options={'time_limit':max(.01,limit),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.x is None:return None,{'status':result.message}
    out=seed.copy();out[:,:2]=result.x[xy]*scale
    out[p.preplaced]=p.specs[p.preplaced]
    # Solve in physical coordinates at finer precision to preserve exact contacts:
    # roundoff is handled separately; never accept overlapping outputs here.
    return out,{'status':result.message,'lp_objective':float(result.fun),'variables':len(obj),'constraints':len(rows)}


def numerical_cleanup(pos,p,edges):
    """Snap active equalities using shared edge values, then recheck hard legality."""
    from .canonical_geometry import canonicalize
    return canonicalize(pos,p)


def contact_tree(seed,p,edges):
    """Minimum geometry-gap spanning tree per public group; replaceable preference."""
    mapping={frozenset((i,j)):(i,j,a) for i,j,a in edges};tree=[]
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==g);parent={int(i):int(i) for i in ids};ranked=[]
        def find(i):
            while parent[i]!=i:i=parent[i]
            return i
        for k,i in enumerate(ids):
            for j in ids[:k]:
                a,b,axis=mapping[frozenset((i,j))];t=1-axis
                gap=abs(seed[b,axis]-seed[a,axis]-seed[a,axis+2])
                tang=max(0.,seed[a,t]-seed[b,t]-seed[b,t+2],seed[b,t]-seed[a,t]-seed[a,t+2])
                ranked.append((gap+2*tang,a,b,axis))
        for _,i,j,a in sorted(ranked):
            if find(i)!=find(j):parent[find(i)]=find(j);tree.append((i,j,a))
    return tree


def refine(raw,inputs,seconds=15.,modes=(0,1),use_contacts=True,shape_steps=0,mip_seconds=0.,reweighted=False):
    begin=time.monotonic();p=prepare_problem(*inputs);objective=Objective(p);trace=[]
    if is_legal(np.asarray(raw),p):incumbent=np.asarray(raw).copy()
    else:incumbent,_=finalize(raw,inputs,min(2.,max(.1,seconds*.15)))
    best=objective(incumbent)
    wh=_dimensions(p,np.asarray(raw)[:,2]/np.asarray(raw)[:,3],True)
    seeds=[np.column_stack((np.asarray(raw)[:,:2],*wh)),incumbent]
    for index,seed in enumerate(seeds):
        for mode in modes:
            edges=topology(seed,p.constraints,mode)
            tree=contact_tree(seed,p,edges) if use_contacts else []
            for weight in ([0.,8.] if tree else [0.]):
                left=seconds-(time.monotonic()-begin)
                if left<.05:break
                candidate,stats=solve_lp(p,seed,edges,contact=weight,contacts=tree if weight else [],limit=min(3.,left))
                if candidate is not None:
                    candidate=numerical_cleanup(candidate,p,edges)
                    legal=is_legal(candidate,p)
                    value=objective(candidate) if legal else None
                    if legal and value<best:incumbent,best=candidate,value
                else:legal=False;value=None
                trace.append({'seed':index,'mode':mode,'contact_weight':weight,'legal':legal,'proxy':value,**stats})
    if shape_steps:
        from .shape_lp import step
        before_shapes=best
        for seed_index,start in enumerate([seeds[0],incumbent.copy()]):
            if seed_index==1 and best<before_shapes-.05:break
            state=start.copy()
            stagnant=0;previous_best=best
            for iteration in range(shape_steps):
                left=seconds-(time.monotonic()-begin)
                if left<.05:break
                edges=topology(state,p.constraints,0)
                candidate,stats=step(p,state,edges,seconds=min(12.,left),elastic=(100000 if reweighted else 100)*(1+iteration/3),iteration=iteration,reweighted=reweighted)
                legal=False;value=None
                if candidate is not None:
                    state=candidate
                    # Shape step may temporarily violate separations. Rebuild
                    # directions each iteration; never accept an illegal state.
                    candidate=numerical_cleanup(candidate,p,edges);legal=is_legal(candidate,p)
                    if legal:
                        value=objective(candidate)
                        if value<best:incumbent,best=candidate,value
                    if not legal or iteration%3==2:
                        fixed,fs=solve_lp(p,candidate,topology(candidate,p.constraints),contact=0.,limit=min(2.,max(.01,seconds-(time.monotonic()-begin))))
                        if fixed is not None:
                            fixed=numerical_cleanup(fixed,p,topology(candidate,p.constraints));fl=is_legal(fixed,p)
                            stats['projection_legal']=fl
                            if fl:
                                stats['projection_proxy']=objective(fixed)
                                if objective(fixed)<best:incumbent,best=fixed,objective(fixed)
                trace.append({'phase':'shape','seed':seed_index,'iteration':iteration,'legal':legal,'proxy':value,**stats})
                if candidate is None:break
                stagnant=stagnant+1 if previous_best-best<.001 else 0
                previous_best=best
                if iteration>=7 and stagnant>=5:break
    if mip_seconds and seconds-(time.monotonic()-begin)>.1:
        from .topology_mip import optimize
        candidate,stats=optimize(p,incumbent,seconds=min(mip_seconds,seconds-(time.monotonic()-begin)),freedom=.15)
        legal=False;value=None
        if candidate is not None:
            candidate=numerical_cleanup(candidate,p,topology(candidate,p.constraints));legal=is_legal(candidate,p)
            if legal:
                value=objective(candidate)
                if value<best:incumbent,best=candidate,value
        trace.append({'phase':'MIP','legal':legal,'proxy':value,**stats})
    return incumbent,{'seconds':time.monotonic()-begin,'proxy':best,'trace':trace}
