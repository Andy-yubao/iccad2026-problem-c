"""Exact rectangle-union/coverage via sorted event cells, differentiable a.e."""
import torch
from .batches import decode

def sorted_coordinates(value):
    # Native deterministic sort backward calls a host scatter path in this build.
    # Detached sorting chooses the same permutation; a one-hot linear map gives
    # the same a.e. derivative without scatter backward or changed precision.
    order=value.detach().argsort(dim=-1,stable=True)
    ids=torch.arange(value.shape[-1],device=value.device)
    permutation=(order[:,:,None]==ids[None,None,:]).to(value.dtype)
    return torch.bmm(permutation,value[:,:,None]).squeeze(-1),order

def measures(state,g):
    p=decode(state,g);valid=g['valid'];lo=p[...,:2];hi=lo+p[...,2:]
    x,_=sorted_coordinates(torch.cat((lo[...,0],hi[...,0]),-1))
    y,yorder=sorted_coordinates(torch.cat((lo[...,1],hi[...,1]),-1))
    dx=x[:,1:]-x[:,:-1];dy=y[:,1:]-y[:,:-1]
    xm=(x[:,1:]+x[:,:-1])/2;ym=(y[:,1:]+y[:,:-1])/2
    ax=(xm[:,:,None]>=lo[:,None,:,0])&(xm[:,:,None]<hi[:,None,:,0])&valid[:,None,:]
    ay=(ym[:,:,None]>=lo[:,None,:,1])&(ym[:,:,None]<hi[:,None,:,1])&valid[:,None,:]
    events=torch.cat((ax.to(torch.int32),-ax.to(torch.int32)),-1)
    events=events.gather(-1,yorder[:,None,:].expand(-1,events.shape[1],-1))
    depth=events.cumsum(-1,dtype=torch.int32)[...,:-1]
    cell=dx[:,:,None]*dy[:,None,:]
    union=(cell*(depth>0)).sum((1,2));overlap_cell=cell*(depth>1)
    overlap_region=overlap_cell.sum((1,2))
    covered=(torch.bmm(overlap_cell,ay.to(p.dtype))*ax.to(p.dtype)).sum(1)
    area=g['area'].clamp_min(1e-12);total=(area*valid).sum(-1).clamp_min(1e-12)
    fraction=(covered/area).clamp(0,1)*valid;count=valid.sum(-1).clamp_min(1)
    bblo=lo.masked_fill(~valid[...,None],float('inf')).amin(1)
    bbhi=hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
    extent=bbhi-bblo;bbox=(extent[:,0]*extent[:,1]).clamp_min(1e-12);excess=(fraction-.1).clamp_min(0)
    return dict(area_excess=(1-union/total).clamp_min(0),overlap_region_fraction=overlap_region/total,
        tail=((excess/(excess+.05)+excess)*valid).sum(-1)/count,
        whitespace=(1-union/bbox).clamp_min(0),covered_fraction=fraction,
        N10=((fraction>.1)&valid).sum(-1).to(p.dtype),
        N10_fraction=((fraction>.1)&valid).sum(-1)/count,max_covered_fraction=fraction.amax(-1),
        union_area=union,bbox_over_A=bbox/total,positions=p)
