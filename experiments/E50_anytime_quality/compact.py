"""Global rigid-component compaction with a reference-free Pareto guard.

Actual group-contact components move together. Every accepted result is hard
legal and has no increase in bbox area, exact wire or discrete soft violations.
The old legal output is retained whenever the guard fails.
"""
import time
import numpy as np
from .backend.frozen.feasible import Objective
from .backend.frozen.solver import hard_valid
from .canonical import canonicalize


def measures(x,p):
    center=x[:,:2]+x[:,2:]/2;wire=0.
    if len(p.b2b):
        i,j=p.b2b[:,:2].astype(int).T;wire+=float((abs(center[i]-center[j]).sum(1)*p.b2b[:,2]).sum())
    if len(p.p2b):
        i,j=p.p2b[:,:2].astype(int).T;wire+=float((abs(p.pins[i]-center[j]).sum(1)*p.p2b[:,2]).sum())
    span=(x[:,:2]+x[:,2:]).max(0)-x[:,:2].min(0)
    return float(span.prod()),wire


def components(x,p):
    parent=np.arange(p.n)
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==g);r=x[ids]
        ov=np.minimum((r[:,:2]+r[:,2:])[:,None],(r[:,:2]+r[:,2:])[None])-np.maximum(r[:,None,:2],r[None,:,:2])
        touch=((ov[:,:,0]>=0)&(ov[:,:,1]>0))|((ov[:,:,0]>0)&(ov[:,:,1]>=0))
        for a,b in zip(*np.nonzero(np.triu(touch,1))):parent[root(int(ids[a]))]=root(int(ids[b]))
    out={}
    for i in range(p.n):out.setdefault(root(i),[]).append(i)
    return [np.array(ids,int) for ids in out.values()]


def median(targets,weights):
    if not len(targets) or weights.sum()<=0:return 0.
    order=np.argsort(targets,kind='stable');cumulative=np.cumsum(weights[order]);return float(targets[order[np.searchsorted(cumulative,.5*cumulative[-1])]])


def polish(seed,p,draft,seconds=.012):
    start=time.perf_counter();x=seed.copy();obj=Objective(p);area,wire=measures(x,p);soft=sum(obj.violations(x));accepted=0;tested=0
    comps=components(x,p);draft_center=draft[:,:2]+draft[:,2:]/2
    # Prioritize displaced global pieces, not individual collision streaks.
    comps.sort(key=lambda ids:-float(np.linalg.norm((x[ids,:2]+x[ids,2:]/2-draft_center[ids]).mean(0))))
    public=[*p.specs[p.fixed|p.preplaced,2:].reshape(-1),*p.specs[p.preplaced,:2].reshape(-1)]
    quantum=2.**(-max([36,*[float(v).as_integer_ratio()[1].bit_length()-1 for v in public]]))
    bi,bj=p.b2b[:,:2].astype(int).T if len(p.b2b) else (np.array([],int),np.array([],int))
    pi,pj=p.p2b[:,:2].astype(int).T if len(p.p2b) else (np.array([],int),np.array([],int))
    for sweep in range(2):
        progress=False
        for ids in comps:
            if time.perf_counter()-start>=seconds:break
            if p.preplaced[ids].any():continue
            inside=np.zeros(p.n,bool);inside[ids]=True;others=np.flatnonzero(~inside)
            for axis in (0,1):
                if time.perf_counter()-start>=seconds:break
                tangent=1-axis;lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0);span=hi-lo
                comp_lo=x[ids,:2].min(0);comp_hi=(x[ids,:2]+x[ids,2:]).max(0)
                lower=-comp_lo[axis];upper=float('inf');box_targets=[]
                if len(others):
                    rest_lo=x[others,:2].min(0);rest_hi=(x[others,:2]+x[others,2:]).max(0)
                    lower=max(lower,rest_hi[axis]-span[axis]-comp_lo[axis]);upper=min(upper,span[axis]+rest_lo[axis]-comp_hi[axis])
                    ov=np.minimum(x[ids,None,tangent]+x[ids,None,tangent+2],x[None,others,tangent]+x[None,others,tangent+2])-np.maximum(x[ids,None,tangent],x[None,others,tangent])
                    same=ov>1e-8
                    before=x[ids,None,axis]+x[ids,None,axis+2]<=x[None,others,axis]+1e-8
                    after=x[ids,None,axis]>=x[None,others,axis]+x[None,others,axis+2]-1e-8
                    if (same&before).any():upper=min(upper,float(np.where(same&before,x[None,others,axis]-x[ids,None,axis]-x[ids,None,axis+2],np.inf).min()))
                    if (same&after).any():lower=max(lower,float(np.where(same&after,x[None,others,axis]+x[None,others,axis+2]-x[ids,None,axis],-np.inf).max()))
                    box_targets=[rest_hi[axis]-comp_hi[axis],rest_lo[axis]-comp_lo[axis]]
                if lower>upper:continue
                centers=x[:,:2]+x[:,2:]/2;targets=[];weights=[]
                if len(bi):
                    out=inside[bi]^inside[bj];a=np.where(inside[bi[out]],bi[out],bj[out]);b=np.where(inside[bi[out]],bj[out],bi[out])
                    targets.extend(centers[b,axis]-centers[a,axis]);weights.extend(p.b2b[out,2]/obj.wire_scale)
                if len(pi):
                    hit=inside[pj];targets.extend(p.pins[pi[hit],axis]-centers[pj[hit],axis]);weights.extend(p.p2b[hit,2]/obj.wire_scale)
                targets=np.asarray(targets);weights=np.asarray(weights)
                combined=np.r_[targets,box_targets];cost=np.r_[weights,np.repeat(.5*span[tangent]/obj.area_scale,len(box_targets))]
                draft_shift=float((draft_center[ids,axis]-centers[ids,axis]).mean())
                proposals=[median(combined,cost),median(targets,weights),draft_shift,*box_targets]
                for delta in dict.fromkeys(float(np.clip(v,lower,upper)) for v in proposals):
                    if time.perf_counter()-start>=seconds:break
                    delta=float(np.clip(np.rint(delta/quantum)*quantum,lower,upper))
                    if abs(delta)<1e-8:continue
                    candidate=x.copy();candidate[ids,axis]+=delta;tested+=1
                    a,w=measures(candidate,p)
                    if a>area or w>wire or (a>=area*(1-1e-7) and w>=wire*(1-1e-7)):continue
                    candidate=canonicalize(candidate,p,tolerance=1e-6)
                    if not hard_valid(candidate,p):continue
                    a,w=measures(candidate,p);v=sum(obj.violations(candidate))
                    if a<=area and w<=wire and v<=soft:
                        x,area,wire,soft=candidate,a,w,v;accepted+=1;progress=True;break
        if not progress or time.perf_counter()-start>=seconds:break
    return x,{'seconds':time.perf_counter()-start,'accepted':accepted,'tested':tested,'components':len(comps),'guard':'hard legal, bbox area<=old, exact wire<=old, soft violations<=old'}
