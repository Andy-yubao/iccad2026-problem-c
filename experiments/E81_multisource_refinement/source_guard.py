"""One-sided protection of source native separation choices; zero at source."""
import torch
from .batches import decode

MARGIN_CAP=.1

def regression(predicted,source,g):
    p=decode(predicted,g);s=decode(source,g)
    pc=p[...,:2]+p[...,2:]/2;sc=s[...,:2]+s[...,2:]/2
    pd=pc[:,:,None]-pc[:,None,:];sd=sc[:,:,None]-sc[:,None,:]
    extent=((s[:,:,None,2:]+s[:,None,:,2:])/2).clamp_min(1e-8)
    normalized=sd.abs()/extent
    x_first=normalized[...,0]>=normalized[...,1]
    source_advantage=(normalized[...,0]-normalized[...,1]).abs()
    signs=torch.where(sd>=0,torch.ones_like(sd),-torch.ones_like(sd))
    chosen=torch.where(x_first,signs[...,0]*pd[...,0]/extent[...,0],signs[...,1]*pd[...,1]/extent[...,1])
    other=torch.where(x_first,pd[...,1].abs()/extent[...,1],pd[...,0].abs()/extent[...,0])
    advantage=chosen-other
    required=source_advantage.clamp_max(MARGIN_CAP)
    n=p.shape[1];pairs=g['valid'][:,:,None]&g['valid'][:,None,:]&~torch.eye(n,device=p.device,dtype=torch.bool)[None]
    pairs=pairs&(g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:])
    penalty=(required-advantage).clamp_min(0).square()*pairs
    mean=penalty.sum((1,2))/pairs.sum((1,2)).clamp_min(1)
    worst=penalty.flatten(1).amax(-1)
    flips=((advantage<0)&pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)
    return mean+.25*worst,flips
