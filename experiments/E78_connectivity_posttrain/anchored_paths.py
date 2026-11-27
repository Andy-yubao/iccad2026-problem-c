"""Public fixed-shape corridor pressure under the actual mode0 pair orders."""
import torch
from .batches import decode

RELAXATIONS=16

def corridor(state,g):
    p=decode(state,g);center=p[...,:2]+p[...,2:]/2;n=p.shape[1]
    delta=center[:,:,None]-center[:,None,:]
    extent=(p[:,:,None,2:]+p[:,None,:,2:])/2
    margin=(delta.abs()-extent)/extent.clamp_min(1e-8)
    horizontal=margin[...,0]>=margin[...,1]
    index=torch.arange(n,device=p.device)
    tie=index[:,None]<index[None,:]
    before=(delta<0)|((delta==0)&tie[None,...,None])
    edges=torch.stack((horizontal&before[...,0],~horizontal&before[...,1]),-1)
    edges=edges&g['valid'][:,:,None,None]&g['valid'][:,None,:,None]
    placed=(~g['mask'][...,0].bool())&g['valid']
    anchor=g['specs'][...,:2]
    lower=torch.where(placed[...,None],anchor,torch.zeros_like(anchor))
    needed=lower
    # Ordinary piecewise gradients through selected max paths and dimensions;
    # no straight-through derivative through discrete axis/order decisions.
    for _ in range(RELAXATIONS):
        proposals=lower[:,:,None,:]+p[:,:,None,2:]
        needed=proposals.masked_fill(~edges,-1e4).amax(1).clamp_min(0)
        lower=torch.where(placed[...,None],anchor,torch.maximum(lower,needed))
    local=g['area'].clamp_min(1e-12).sqrt()
    deficit=((needed-anchor).clamp_min(0)/local[...,None])*placed[...,None]
    squared=deficit.square()
    mean=squared.sum((1,2))/(2*placed.sum(1)).clamp_min(1)
    worst=squared.flatten(1).amax(-1)
    return mean+.25*worst,deficit
