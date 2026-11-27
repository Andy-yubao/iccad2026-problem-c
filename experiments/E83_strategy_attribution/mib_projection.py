"""Tie compatible public MIB log-aspects before the original frame mapping."""
import torch
from .frame import contain

def coherent_shapes(state,g,frame):
    n=state.shape[1];eye=torch.eye(n,device=state.device,dtype=state.dtype)[None]
    active=(g['constraints'][...,2]>0)&g['valid']
    members=(g['mib']>0).to(state.dtype)+eye*active[...,None]
    members=members*g['valid'][:,:,None]*g['valid'][:,None,:]
    count=members.sum(-1).clamp_min(1)
    log_area=g['area'].clamp_min(1e-12).log()
    maximum=log_area[:,None].expand(-1,n,-1).masked_fill(members==0,-1e4).amax(-1)
    minimum=log_area[:,None].expand(-1,n,-1).masked_fill(members==0,1e4).amin(-1)
    eligible=active&((maximum-minimum).abs()<2e-6)
    locked=(~g['mask'][...,2].bool())&g['valid']
    locked_count=(members*locked[:,None]).sum(-1)
    ratio=(members@state[...,2,None]).squeeze(-1)/count
    locked_ratio=(members@(g['known'][...,2]*locked)[...,None]).squeeze(-1)/locked_count.clamp_min(1)
    known=g['known'][...,2][:,None].expand(-1,n,-1)
    select=(members>0)&locked[:,None]
    locked_spread=known.masked_fill(~select,-1e4).amax(-1)-known.masked_fill(~select,1e4).amin(-1)
    eligible=eligible&((locked_count==0)|(locked_spread.abs()<2e-6))
    ratio=torch.where(locked_count>0,locked_ratio,ratio)
    ratio=torch.where(eligible&~locked,ratio,state[...,2])
    return contain(torch.cat((state[...,:2],ratio[...,None]),-1),g,frame)
