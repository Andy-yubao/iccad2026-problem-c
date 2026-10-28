"""Exact public-area-normalized pairwise intersection, matching raw diagnostics."""
import torch
from .batches import decode
def fraction(state,g):
    p=decode(state,g);lo=p[...,:2];hi=lo+p[...,2:];n=p.shape[1]
    inter=(torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(lo[:,:,None],lo[:,None,:])).clamp_min(0)
    ids=torch.arange(n,device=state.device)
    pairs=(ids[:,None]<ids[None,:])[None]&g['valid'][:,:,None]&g['valid'][:,None,:]
    area=(g['area']*g['valid']).sum(-1).clamp_min(1e-12)
    return ((inter[...,0]*inter[...,1])*pairs).sum((1,2))/area
