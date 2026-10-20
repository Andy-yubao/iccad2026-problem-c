"""Near-draft greedy legalization, preserving predicted dimensions and geometry."""
import numpy as np
from .backend.frozen.packing import _dimensions
from .canonical import canonicalize


def seed(p,draft,order_mode=0,preplaced_only=False,preserve_centers=False):
    widths,heights=_dimensions(p,draft[:,2]/draft[:,3],True)
    result=np.column_stack((draft[:,:2],widths,heights));result[p.preplaced]=p.specs[p.preplaced]
    if preserve_centers:
        result[~p.preplaced,:2]=np.maximum(0,draft[~p.preplaced,:2]+draft[~p.preplaced,2:]/2-result[~p.preplaced,2:]/2)
    placed=list(np.flatnonzero(p.preplaced));ids=np.flatnonzero(~p.preplaced)
    center=draft[:,:2]+draft[:,2:]/2;codes=p.constraints[:,4].astype(int)
    if order_mode==0:
        order=sorted(ids,key=lambda i:(0 if codes[i] else 1,-p.areas[i],i))
    elif order_mode==1:order=sorted(ids,key=lambda i:(round(center[i,1]/(p.scale*.1)),center[i,0],i))
    else:order=sorted(ids,key=lambda i:(round(center[i,0]/(p.scale*.1)),center[i,1],i))
    target_hi=(draft[:,:2]+draft[:,2:]).max(0)
    for i in order:
        w,h=result[i,2:];desired=np.maximum(result[i,:2],0.)
        if not placed:result[i,:2]=desired;placed.append(i);continue
        obstacles=result[placed];lo=obstacles[:,:2];hi=lo+obstacles[:,2:]
        overlap=np.minimum(desired+[w,h],hi)-np.maximum(desired,lo)
        if not (overlap>1e-8).all(1).any():
            result[i,:2]=desired
            if not preplaced_only:placed.append(i)
            continue
        xs=np.unique(np.r_[desired[0],0.,hi[:,0],lo[:,0]-w]);xs=xs[xs>=0]
        cross=(xs[:,None]<hi[None,:,0]-1e-10)&(xs[:,None]+w>lo[None,:,0]+1e-10)
        up=np.full(len(xs),desired[1]);down=up.copy();active=np.ones(len(xs),dtype=bool)
        for _ in range(len(placed)+1):
            hit=cross&(up[:,None]<hi[None,:,1]-1e-10)&(up[:,None]+h>lo[None,:,1]+1e-10)
            if not hit.any():break
            up=np.maximum(up,np.where(hit,hi[None,:,1],0).max(1))
        for _ in range(len(placed)+1):
            hit=cross&(down[:,None]<hi[None,:,1]-1e-10)&(down[:,None]+h>lo[None,:,1]+1e-10)
            if not hit.any():break
            down=np.minimum(down,np.where(hit,lo[None,:,1]-h,np.inf).min(1))
        candidates=np.column_stack((np.tile(xs,2),np.r_[up,down]));candidates=candidates[candidates[:,1]>=0]
        delta=(candidates-desired)/np.maximum([w,h],1e-8);cost=(delta**2).sum(1)
        # Soft public boundary preferences; no reference coordinates are read.
        if codes[i]&1:cost+=.3*(candidates[:,0]/max(w,1e-8))**2
        if codes[i]&8:cost+=.3*(candidates[:,1]/max(h,1e-8))**2
        if codes[i]&2:cost+=.3*((candidates[:,0]+w-target_hi[0])/max(w,1e-8))**2
        if codes[i]&4:cost+=.3*((candidates[:,1]+h-target_hi[1])/max(h,1e-8))**2
        group=p.constraints[i,3]
        if group>0:
            members=np.flatnonzero(p.constraints[placed,3]==group)
            if len(members):
                gl,gh=lo[members],hi[members]
                inter=np.minimum(candidates[:,None]+[w,h],gh)-np.maximum(candidates[:,None],gl)
                touches=((np.abs(inter[:,:,0])<1e-7)&(inter[:,:,1]>1e-7))|((np.abs(inter[:,:,1])<1e-7)&(inter[:,:,0]>1e-7))
                cost+=.15*(~touches.any(1))
        result[i,:2]=candidates[np.argmin(cost)]
        if not preplaced_only:placed.append(i)
    return canonicalize(result,p)
