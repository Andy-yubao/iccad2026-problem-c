"""Current-public-layout summaries only; no teacher or proposed state update."""
import torch
from .original_model import decode

FEATURE_NAMES=(
    'bbox_center_x','bbox_center_y','width_local','height_local','log_aspect',
    'overlap_count_log','overlap_load_log','overlap_max_log',
    'travel_L','travel_R','travel_T','travel_B',
    'probe025_L','probe025_R','probe025_T','probe025_B',
    'has_group_peer','group_peer_count_log',
    'near1_x','near1_y','near1_gap','near1_valid',
    'near2_x','near2_y','near2_gap','near2_valid',
    'near3_x','near3_y','near3_gap','near3_valid',
    'group_center_x','group_center_y','group_spread_x','group_spread_y',
    'requires_L','requires_R','requires_T','requires_B',
    'boundary_L','boundary_R','boundary_T','boundary_B',
    'has_corner','corner_x','corner_y')
_BITS={}
TRAVEL_EPSILON=.1


@torch.no_grad()
def bounded_travel(p,g,bblo=None,bbhi=None):
    """Exact first forbidden interval of pair penetration; output physical LRTB.

    rho=max(0,min(axis slacks))/pair_scale. With one axis translated, the
    forbidden set rho>rho_at_start+.1 is an open interval, or empty. Its first
    positive entry limits the continuous prefix, even if the endpoint is clear.
    """
    lo=p[...,:2];wh=p[...,2:];hi=lo+wh;center=lo+wh/2
    valid=g['valid'];local=g['area'].clamp_min(1e-12).sqrt().clamp_min(1e-6)
    n=p.shape[1];ids=torch.arange(n,device=p.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
    if bblo is None:
        any_valid=valid.any(-1)
        bblo=lo.masked_fill(~valid[...,None],float('inf')).amin(1)
        bbhi=hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
        bblo=torch.where(any_valid[:,None],bblo,torch.zeros_like(bblo))
        bbhi=torch.where(any_valid[:,None],bbhi,torch.ones_like(bbhi))
    delta=center[:,:,None]-center[:,None,:]
    half=(wh[:,:,None]+wh[:,None,:])/2
    slack=half-delta.abs()
    scale=(local[:,:,None]+local[:,None,:])/2
    baseline=slack.amin(-1).clamp_min(0)
    threshold=baseline+TRAVEL_EPSILON*scale
    boundary=torch.stack((lo[...,0]-bblo[:,None,0],bbhi[:,None,0]-hi[...,0],
        bbhi[:,None,1]-hi[...,1],lo[...,1]-bblo[:,None,1]),-1).clamp_min(0)
    side=torch.stack((wh[...,0],wh[...,0],wh[...,1],wh[...,1]),-1)
    cap=torch.minimum(boundary,side)
    distances=[]
    for k,(axis,direction) in enumerate(((0,-1.),(0,1.),(1,1.),(1,-1.))):
        radius=half[...,axis]-threshold
        first=(-radius-delta[...,axis])/direction
        last=(radius-delta[...,axis])/direction
        enter=torch.minimum(first,last);leave=torch.maximum(first,last)
        forbidden=pairs&(radius>0)&(slack[...,1-axis]>threshold)&(leave>enter.clamp_min(0))
        stop=enter.clamp_min(0).masked_fill(~forbidden,float('inf')).amin(-1)
        distances.append(torch.minimum(cap[...,k],stop))
    return torch.stack(distances,-1)*valid[...,None]


def pair_features(state,g):
    """Exactly the parent's18 bias inputs, also reused for relation messages."""
    p=decode(state,g);wh=p[...,2:];center=p[...,:2]+wh/2
    delta=center[:,:,None]-center[:,None,:]
    raw=state[:,:,None,:2]-state[:,None,:,:2]
    local=g['area'].clamp_min(1e-12).sqrt()
    scale=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
    n=state.shape[1];normalized_wh=wh/local[...,None]
    gap=(delta.abs()-(wh[:,:,None]+wh[:,None,:])/2)/scale[...,None]
    flags=g['constraints'][...,:2].to(state.dtype)
    dimensions=torch.cat((normalized_wh[:,:,None].expand(-1,-1,n,-1),normalized_wh[:,None,:].expand(-1,n,-1,-1)),-1)
    anchors=torch.cat((flags[:,:,None].expand(-1,-1,n,-1),flags[:,None,:].expand(-1,n,-1,-1)),-1)
    pair=torch.cat((raw,raw.abs(),g['adjacency'][...,None],(g['group']>0)[...,None],dimensions,gap,(-gap).clamp_min(0),anchors),-1)
    return pair,p


@torch.no_grad()
def module_features(p,g):
    """45 detached node features; only E142's direction channels change."""
    p=p.detach();lo=p[...,:2];wh=p[...,2:];hi=lo+wh;center=lo+wh/2
    valid=g['valid'];area=g['area'].clamp_min(1e-12);local=area.sqrt().clamp_min(1e-6)
    b,n,_=p.shape;index=torch.arange(n,device=p.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(index[:,None]!=index[None,:])[None]
    scale=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
    intersection=(torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(lo[:,:,None],lo[:,None,:])).clamp_min(0)
    covered=intersection.prod(-1)*pairs
    load=covered.sum(-1)/area
    colliding=(intersection>1e-6*scale[...,None]).all(-1)&pairs
    any_valid=valid.any(-1)
    bblo=lo.masked_fill(~valid[...,None],float('inf')).amin(1)
    bbhi=hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
    bblo=torch.where(any_valid[:,None],bblo,torch.zeros_like(bblo))
    bbhi=torch.where(any_valid[:,None],bbhi,torch.ones_like(bbhi))
    bbox_center=(center-bblo[:,None])/(bbhi-bblo).clamp_min(1e-6)[:,None]
    base=torch.cat((bbox_center,wh/local[...,None],(wh[...,0]/wh[...,1].clamp_min(1e-12)).clamp_min(1e-12).log()[...,None]),-1).asinh()
    overlap=torch.stack((colliding.sum(-1).to(p.dtype).log1p(),load.log1p(),(covered.amax(-1)/area).log1p()),-1)
    travel=(bounded_travel(p,g,bblo,bbhi)/local[...,None]).asinh()
    probes=[]
    for distance in (.25,):
        for dx,dy in ((-1.,0.),(1.,0.),(0.,1.),(0.,-1.)):
            offset=torch.stack((distance*dx*local,distance*dy*local),-1)
            trial=(torch.minimum((hi+offset)[:,:,None],hi[:,None,:])-torch.maximum((lo+offset)[:,:,None],lo[:,None,:])).clamp_min(0)
            trial_load=(trial.prod(-1)*pairs).sum(-1)/area
            probes.append((trial_load.log1p()-load.log1p()).asinh())
    probes=torch.stack(probes,-1)
    same=(g['group']>0)&pairs;count=same.sum(-1);has=count>0
    delta=center[:,None,:]-center[:,:,None]
    free_gap=(torch.maximum(lo[:,:,None],lo[:,None,:])-torch.minimum(hi[:,:,None],hi[:,None,:])).clamp_min(0)
    gap=free_gap.square().sum(-1).sqrt()/scale
    # Tiny center-distance tie breaker for overlapping/abutting candidates.
    rank=(gap+1e-6*delta.square().sum(-1).sqrt()/scale).masked_fill(~same,float('inf'))
    k=min(3,n);_,nearest=rank.topk(k,dim=-1,largest=False,sorted=True)
    nearby=[]
    for j in range(3):
        if j<k:
            ids=nearest[...,j:j+1];found=same.gather(2,ids).squeeze(-1)
            vec=(delta/scale[...,None]).gather(2,ids[...,None].expand(-1,-1,-1,2)).squeeze(2)
            dist=gap.gather(2,ids).squeeze(-1)
            nearby.append(torch.cat((vec.asinh()*found[...,None],dist.asinh()[...,None]*found[...,None],found.to(p.dtype)[...,None]),-1))
        else:nearby.append(torch.zeros((b,n,4),device=p.device,dtype=p.dtype))
    weights=same.to(p.dtype)*area[:,None,:];mass=weights.sum(-1).clamp_min(1e-12)
    group_center=torch.bmm(weights,center)/mass[...,None]
    second=torch.bmm(weights,center.square())/mass[...,None]
    spread=(second-group_center.square()).clamp_min(0).sqrt()/local[...,None]
    group=torch.cat((has.to(p.dtype)[...,None],count.to(p.dtype).log1p()[...,None],*nearby,
        ((group_center-center)/local[...,None]).asinh()*has[...,None],spread.asinh()*has[...,None]),-1)
    key=(str(p.device),g['constraints'].dtype)
    if key not in _BITS:_BITS[key]=torch.tensor((1,2,4,8),device=p.device,dtype=g['constraints'].dtype)
    bits=_BITS[key]
    required=(g['constraints'][...,4,None]&bits)!=0
    distances=torch.stack((lo[...,0]-bblo[:,None,0],bbhi[:,None,0]-hi[...,0],bbhi[:,None,1]-hi[...,1],lo[...,1]-bblo[:,None,1]),-1)/local[...,None]
    corner=(required[...,:2].sum(-1)==1)&(required[...,2:].sum(-1)==1)
    corner_vector=torch.stack((torch.where(required[...,0],-distances[...,0],distances[...,1]),
        torch.where(required[...,2],distances[...,2],-distances[...,3])),-1)*corner[...,None]
    boundary=torch.cat((required.to(p.dtype),distances.asinh(),corner.to(p.dtype)[...,None],corner_vector.asinh()),-1)
    out=torch.cat((base,overlap,travel,probes,group,boundary),-1)
    return out*valid[...,None]
