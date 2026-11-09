"""Read-only no-direct-port relative structure diagnostics; not training Loss."""
import torch
from .base_model import decode

@torch.no_grad()
def diagnostics(state,g):
    p=decode(state,g);r=g['reference'];pc=p[...,:2]+p[...,2:]/2;rc=r[...,:2]+r[...,2:]/2
    pd=pc[:,:,None]-pc[:,None,:];rd=rc[:,:,None]-rc[:,None,:]
    local=g['area'].clamp_min(1e-12).sqrt();scale=(local[:,:,None]+local[:,None,:])/2
    rextent=(r[:,:,None,2:]+r[:,None,:,2:])/2;rm=rd.abs()-rextent
    n=state.shape[1];ids=torch.arange(n,device=state.device)
    noport=g['features'][...,14]==0
    pairs=g['valid'][:,:,None]&g['valid'][:,None,:]&(ids[:,None]<ids[None,:])[None]
    pairs=pairs&noport[:,:,None]&noport[:,None,:]
    near=pairs&(rm.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)
    count=near.sum((1,2));error=((pd-rd)/scale.clamp_min(1e-8)[...,None]).abs().mean(-1)
    axis=rm[...,0]>=rm[...,1];desired=torch.where(axis,rd[...,0],rd[...,1]);actual=torch.where(axis,pd[...,0],pd[...,1])
    clear=near&(desired.abs()>.25*scale);flips=(desired*actual<0)&clear
    return dict(noport_near_pair_count=count,
        noport_near_relative_error=(error*near).sum((1,2))/count.clamp_min(1),
        noport_order_clear_count=clear.sum((1,2)),noport_order_flip_count=flips.sum((1,2)))
