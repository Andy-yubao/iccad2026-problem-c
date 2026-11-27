"""C3 fixed physical intent plus public boundary/group-contact task, no reference."""
import torch
import torch.nn.functional as F
from .candidate3 import per_case as c3_per_case
from .nn85.base_model import decode

_BITS={}
def raw_terms(state,g,target,usable):
    out=c3_per_case(state,g,target,usable);p=decode(state,g);valid=g['valid'];local=g['area'].clamp_min(1e-12).sqrt()
    lo=p[...,:2];hi=lo+p[...,2:]
    minimum=lo.masked_fill(~valid[...,None],1e4).amin(1);maximum=hi.masked_fill(~valid[...,None],-1e4).amax(1)
    side=torch.stack((lo[...,0]-minimum[:,None,0],maximum[:,None,0]-hi[...,0],maximum[:,None,1]-hi[...,1],lo[...,1]-minimum[:,None,1]),-1)/local[...,None]
    if state.device not in _BITS:_BITS[state.device]=torch.tensor([1,2,4,8],device=state.device)
    required=((g['constraints'][...,4,None]&_BITS[state.device])!=0)&valid[...,None]
    boundary=torch.where(required,F.smooth_l1_loss(side,torch.zeros_like(side),beta=.1,reduction='none'),0.).amax(-1).sum(-1)
    center=lo+p[...,2:]/2
    # Teacher contact tree is directed a->b along the teacher-native axis.
    delta=center[:,None]-center[:,:,None];extent=(p[:,None,:,2:]+p[:,:,None,2:])/2
    axis=g['label_contact_x'];normal=torch.where(axis,delta[...,0]/extent[...,0].clamp_min(1e-8),delta[...,1]/extent[...,1].clamp_min(1e-8)).clamp(-8,8)
    tangential=torch.where(axis,extent[...,1]-delta[...,1].abs(),extent[...,0]-delta[...,0].abs())
    pair_local=(local[:,None]+local[:,:,None])/2
    error=F.smooth_l1_loss(normal,torch.ones_like(normal),beta=.1,reduction='none')+torch.relu(.01-tangential/pair_local.clamp_min(1e-8)).square()
    contact=(error*g['label_contact_mask']).sum((1,2));denom=g['public_soft_count'].clamp_min(1)
    out['public_boundary']=boundary/denom;out['public_group_contact']=contact/denom
    out['public_soft']=out['public_boundary']+out['public_group_contact']
    return out
