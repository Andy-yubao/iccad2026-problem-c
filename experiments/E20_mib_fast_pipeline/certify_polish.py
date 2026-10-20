"""Cheap rigid moves with a public-input certificate of neutral-Q non-regression."""
import time
import math
import numpy as np
from .solver import raw_measures
from .backend.frozen.feasible import Objective


def polish(x,p,seconds=.01,max_accept=4,push=False):
    start=time.perf_counter();x=x.copy();obj=Objective(p);area,wire=raw_measures(x,p);soft=sum(obj.violations(x));accepted=0;tested=0
    def try_move(ids,shift):
        nonlocal tested
        ids=np.asarray(ids,dtype=int)
        if p.preplaced[ids].any():return None
        moved=x[ids].copy();moved[:,:2]+=shift
        if (moved[:,:2]<0).any():return None
        while True:
            other=np.ones(p.n,dtype=bool);other[ids]=False;rest=x[other]
            overlaps=np.minimum(moved[:,None,:2]+moved[:,None,2:],rest[None,:,:2]+rest[None,:,2:])-np.maximum(moved[:,None,:2],rest[None,:,:2])
            blocked=(overlaps>1e-7).all(-1).any(0)
            if not blocked.any():break
            if not push or time.perf_counter()-start>=seconds:return None
            added=np.flatnonzero(other)[blocked]
            if p.preplaced[added].any():return None
            ids=np.r_[ids,added];moved=x[ids].copy();moved[:,:2]+=shift
            if (moved[:,:2]<0).any():return None
        candidate=x.copy();candidate[ids]=moved;tested+=1
        if push:
            from .canonical import canonicalize
            from .backend.frozen.solver import hard_valid
            candidate=canonicalize(candidate,p,tolerance=1e-6)
            if not hard_valid(candidate,p):return None
        v=sum(obj.violations(candidate))
        if v>=soft:return None
        a,w=raw_measures(candidate,p)
        factor=max(1.,a/max(area,1e-15),w/max(wire,1e-15))*math.exp(2.*(v-soft)/obj.nsoft)
        return (factor,candidate,a,w,v) if factor<=1.-1e-10 else None
    while accepted<max_accept and time.perf_counter()-start<seconds and soft:
        best=None;proposals=[];lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0)
        for i in np.flatnonzero(obj.codes):
            if p.preplaced[i]:continue
            shift=np.zeros(2);code=obj.codes[i]
            if code&1:shift[0]=lo[0]-x[i,0]
            elif code&2:shift[0]=hi[0]-x[i,0]-x[i,2]
            if code&8:shift[1]=lo[1]-x[i,1]
            elif code&4:shift[1]=hi[1]-x[i,1]-x[i,3]
            if np.abs(shift).max()>1e-8:proposals.append(([int(i)],shift))
        for group in obj.groups:
            lower=x[group,:2];upper=lower+x[group,2:];ov=np.minimum(upper[:,None],upper[None])-np.maximum(lower[:,None],lower[None])
            touch=((ov[:,:,0]>=0)&(ov[:,:,1]>0))|((ov[:,:,0]>0)&(ov[:,:,1]>=0))
            unseen=set(range(len(group)));components=[]
            while unseen:
                todo=[unseen.pop()];component=[]
                while todo:
                    i=todo.pop();component.append(i);neighbors=unseen.intersection(np.flatnonzero(touch[i]).tolist());unseen-=neighbors;todo+=list(neighbors)
                components.append(group[component])
            if len(components)<2:continue
            for ids in components:
                if p.preplaced[ids].any():continue
                others=group[~np.isin(group,ids)]
                for i in ids:
                    for j in others:
                        for axis in [0,1]:
                            t=1-axis;eps=.001*min(x[i,t+2],x[j,t+2])
                            orthogonal=float(np.clip(0,x[j,t]-x[i,t]-x[i,t+2]+eps,x[j,t]+x[j,t+2]-x[i,t]-eps))
                            for shift_axis in [x[j,axis]-x[i,axis]-x[i,axis+2],x[j,axis]+x[j,axis+2]-x[i,axis]]:
                                shift=np.zeros(2);shift[axis]=shift_axis;shift[t]=orthogonal
                                proposals.append((ids,shift))
        for ids,shift in proposals:
            if time.perf_counter()-start>=seconds:break
            candidate=try_move(ids,shift)
            if candidate is not None and (best is None or candidate[0]<best[0]):best=candidate
        if best is None:break
        _,x,area,wire,soft=best;accepted+=1
    return x,{'seconds':time.perf_counter()-start,'accepted':accepted,'tested':tested}
