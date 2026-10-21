"""Experiment-local differentiable occupancy and exact diagnostic geometry."""
import numpy as np
import torch
from .batches import decode

EXTRA_WEIGHTS={'overlap_area':1.,'crowding':.25,'void':.25,'compactness':.15}
GRID_SIDE=32


def components(state,g):
    p=decode(state,g);valid=g['valid'];area=g['area']*valid
    xy=p[...,:2];wh=p[...,2:];hi=xy+wh
    inter=(torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(xy[:,:,None],xy[:,None,:])).clamp_min(0).prod(-1)
    n=state.shape[1]
    pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
    # Immutable preplaced pairs have no trainable remedy and are excluded.
    pairs=pairs&(g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:])
    total=area.sum(1).clamp_min(1e-8)
    overlap=(inter*pairs).sum((1,2))/(2*total)
    load=(inter*pairs/area[:,:,None].clamp_min(1e-8)).sum(2)
    crowding=((load-.5).clamp_min(0).square()*valid).sum(1)/valid.sum(1).clamp_min(1)
    lo=xy.masked_fill(~valid[...,None],1e4).amin(1)
    upper=hi.masked_fill(~valid[...,None],-1e4).amax(1)
    span=(upper-lo).clamp_min(1e-6);bbox=span.prod(-1)
    compactness=(bbox/total).clamp_min(1).log().square()
    u=(torch.arange(GRID_SIDE,device=state.device,dtype=state.dtype)+.5)/GRID_SIDE
    grid=torch.stack(torch.meshgrid(u,u,indexing='ij'),-1).reshape(-1,2)
    points=lo[:,None,:]+grid[None]*span[:,None,:]
    # Temperature scales with each rectangle; differentiable bbox follows draft.
    temperature=(wh*.04).clamp_min(.002)
    left=torch.sigmoid((points[:,None]-xy[:,:,None])/temperature[:,:,None])
    right=torch.sigmoid((hi[:,:,None]-points[:,None])/temperature[:,:,None])
    coverage=(left*right).prod(-1)*valid[...,None]
    union=-torch.expm1(torch.log1p(-coverage.clamp(max=1-1e-6)).sum(1))
    void=(1-union).mean(1)
    values={'overlap_area':overlap,'crowding':crowding,'void':void,'compactness':compactness}
    values['occupancy']=sum(EXTRA_WEIGHTS[k]*v for k,v in values.items())
    return values


def exact_diagnostics(positions):
    """Axis sweep integrates multiplicity; no raster approximation in reports."""
    p=np.asarray(positions,dtype=float);xy=p[:,:2];hi=xy+p[:,2:];n=len(p)
    if not n or not np.isfinite(p).all() or (p[:,2:]<=0).any():raise ValueError('Invalid rectangles')
    hist=np.zeros(n+1);xs=np.unique(np.r_[xy[:,0],hi[:,0]])
    for x0,x1 in zip(xs[:-1],xs[1:]):
        ids=np.flatnonzero((xy[:,0]<x1)&(hi[:,0]>x0))
        events={}
        for i in ids:
            events[xy[i,1]]=events.get(xy[i,1],0)+1
            events[hi[i,1]]=events.get(hi[i,1],0)-1
        count=0;last=None
        for y,delta in sorted(events.items()):
            if last is not None:hist[count]+=(x1-x0)*(y-last)
            count+=delta;last=y
    bbox=float(np.prod(hi.max(0)-xy.min(0)));union=float(hist[1:].sum())
    inter=np.minimum(hi[:,None],hi[None,:])-np.maximum(xy[:,None],xy[None,:])
    graph=(inter>1e-8).all(-1);np.fill_diagonal(graph,False)
    seen=set();sizes=[]
    for i in range(n):
        if i in seen:continue
        stack=[i];seen.add(i);size=0
        while stack:
            j=stack.pop();size+=1
            for k in np.flatnonzero(graph[j]):
                k=int(k)
                if k not in seen:seen.add(k);stack.append(k)
        sizes.append(size)
    total=float(p[:,2:].prod(1).sum())
    return {'bbox_area':bbox,'union_area':union,'empty_fraction':max(0.,1-union/bbox),
            'overlap_excess_area_fraction':max(0.,(total-union)/total),
            'multi_covered_area_fraction':float(hist[2:].sum()/bbox),
            'triple_covered_area_fraction':float(hist[3:].sum()/bbox),
            'max_multiplicity':int(np.flatnonzero(hist>1e-10)[-1]) if (hist>1e-10).any() else 0,
            'overlap_pair_count':int(graph.sum()//2),'overlap_block_fraction':float(graph.any(1).mean()),
            'max_conflict_degree':int(graph.sum(1).max()),'largest_conflict_component':max(sizes),
            'conflict_components_size_ge3':sum(s>=3 for s in sizes)}
