"""Public geometry task for C8; full ports enter Loss only, never NN tokens.
Corridor pressure/penetration/boundary/MIB formulas independently adapted from
E85 public geometry Loss; no cross-experiment imports or free reference targets.
"""
import torch
from .candidate7 import per_case as teacher_loss
from .nn85.base_model import decode
BITS={}

def raw_terms(state,g,target,usable):
    teacher=teacher_loss(state,g,target,usable)['total'];p=decode(state,g);valid=g['valid'];n=p.shape[1]
    center=p[...,:2]+p[...,2:]/2;wh=p[...,2:];local=g['area'].clamp_min(1e-12).sqrt()
    delta=center[:,:,None]-center[:,None,:];extent=(wh[:,:,None]+wh[:,None,:])/2
    ids=torch.arange(n,device=state.device);pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]<ids[None,:])[None]
    mutable=g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:];pairs=pairs&mutable
    lo=p[...,:2];hi=lo+wh;inter=(torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(lo[:,:,None],lo[:,None,:])).clamp_min(0)
    total_area=(g['area']*valid).sum(-1).clamp_min(1e-12)
    intersection=((inter[...,0]*inter[...,1])*pairs).sum((1,2))/total_area
    margin=delta.abs()-extent;nearest=torch.where(margin[...,0]>=margin[...,1],margin[...,0],margin[...,1]);scale=(local[:,:,None]+local[:,None,:])/2
    penetration=(((-nearest).clamp_min(0)/scale.clamp_min(1e-8)).square()*pairs).sum((1,2))/valid.sum(-1).clamp_min(1)
    lower=lo.masked_fill(~valid[...,None],1e4).amin(1);upper=hi.masked_fill(~valid[...,None],-1e4).amax(1)
    side=torch.stack((p[...,0]-lower[:,None,0],upper[:,None,0]-hi[...,0],upper[:,None,1]-hi[...,1],p[...,1]-lower[:,None,1]),-1)
    if state.device not in BITS:BITS[state.device]=torch.tensor([1,2,4,8],device=state.device)
    required=((g['constraints'][...,4,None]&BITS[state.device])!=0)&valid[...,None]
    boundary_error=torch.nn.functional.smooth_l1_loss(side/local[...,None],torch.zeros_like(side),beta=.1,reduction='none')
    boundary=(boundary_error*required).sum((1,2))/required.sum((1,2)).clamp_min(1)
    compatible=(g['area'].clamp_min(1e-12).log()[:,:,None]-g['area'].clamp_min(1e-12).log()[:,None,:]).abs()<.019
    free_shape=g['mask'][...,2].bool()[:,:,None]|g['mask'][...,2].bool()[:,None,:]
    mib=(g['mib']>0)&compatible&pairs&free_shape;logwh=wh.clamp_min(1e-8).log()
    mib_error=(logwh[:,:,None]-logwh[:,None,:]).square().mean(-1);mib_loss=(mib_error*mib).sum((1,2))/mib.sum((1,2)).clamp_min(1)
    physics=intersection+penetration+.25*boundary+.1*mib_loss
    # Fixed16 native-order longest-path relaxations: public anchored feasibility pressure.
    native=margin/extent.clamp_min(1e-8);horizontal=native[...,0]>=native[...,1]
    tie=ids[:,None]<ids[None,:];before=(delta<0)|((delta==0)&tie[None,...,None])
    edges=torch.stack((horizontal&before[...,0],~horizontal&before[...,1]),-1)&valid[:,:,None,None]&valid[:,None,:,None]
    placed=(~g['mask'][...,0].bool())&valid;anchor=g['specs'][...,:2]
    position=torch.where(placed[...,None],anchor,torch.zeros_like(anchor));needed=position
    for _ in range(16):
        proposal=position[:,:,None]+wh[:,:,None]
        needed=proposal.masked_fill(~edges,-1e4).amax(1).clamp_min(0)
        position=torch.where(placed[...,None],anchor,torch.maximum(position,needed))
    deficit=((needed-anchor).clamp_min(0)/local[...,None])*placed[...,None];square=deficit.square()
    corridor_raw=square.sum((1,2))/(2*placed.sum(1)).clamp_min(1)+.25*square.flatten(1).amax(-1)
    corridor=corridor_raw.log1p()
    span=upper-lower;bbox=span[:,0]*span[:,1];bbox_ratio=bbox/total_area
    b2b=.5*(delta.abs().sum(-1)*g['b2b_weights']).sum((1,2))
    pin_delta=center[:,:,None]-g['port_xy'][:,None]
    p2b=(pin_delta.abs().sum(-1)*g['p2b_weights']).sum((1,2))
    net_weight=.5*g['b2b_weights'].sum((1,2))+g['p2b_weights'].sum((1,2))
    wire_ratio=(b2b+p2b)/(net_weight*total_area.sqrt()).clamp_min(1e-8)
    quality=.5*bbox_ratio.log1p()+.5*wire_ratio.log1p()
    return dict(teacher=teacher,physics=physics,corridor=corridor,quality=quality,intersection=intersection,penetration=penetration,boundary=boundary,mib=mib_loss,bbox_ratio=bbox_ratio,wire_ratio=wire_ratio,corridor_raw=corridor_raw)
