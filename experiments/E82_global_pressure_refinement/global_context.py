"""Public source-draft interval pressures; no reference, labels or solver calls."""
import torch
from .base_model import decode

@torch.no_grad()
def interval_context(base,g,frame):
    p=decode(base,g);xy=p[...,:2];wh=p[...,2:];center=xy+wh/2;n=p.shape[1]
    delta=center[:,:,None]-center[:,None,:];extent=(wh[:,:,None]+wh[:,None,:])/2
    margin=(delta.abs()-extent)/extent.clamp_min(1e-8)
    horizontal=margin[...,0]>=margin[...,1]
    index=torch.arange(n,device=p.device);tie=index[:,None]<index[None,:]
    before=(delta<0)|((delta==0)&tie[None,...,None])
    edges=torch.stack((horizontal&before[...,0],~horizontal&before[...,1]),-1)
    edges=edges&g['valid'][:,:,None,None]&g['valid'][:,None,:,None]
    placed=(~g['mask'][...,0].bool())&g['valid'];anchor=g['specs'][...,:2]
    floor=frame[0][:,None].expand_as(xy);ceiling=frame[1][:,None]-wh
    lower=torch.where(placed[...,None],anchor,floor);upper=torch.where(placed[...,None],anchor,ceiling)
    for _ in range(16):
        incoming=(lower[:,:,None]+wh[:,:,None]).masked_fill(~edges,-1e4).amax(1)
        outgoing=(upper[:,None]-wh[:,:,None]).masked_fill(~edges,1e4).amin(2)
        need_lower=torch.maximum(floor,incoming);need_upper=torch.minimum(ceiling,outgoing)
        lower=torch.where(placed[...,None],anchor,torch.maximum(lower,need_lower))
        upper=torch.where(placed[...,None],anchor,torch.minimum(upper,need_upper))
    local=g['area'].clamp_min(1e-12).sqrt()[...,None]
    # Positive values are violations; negative values are remaining directional slack.
    features=torch.cat(((need_lower-xy)/local,(xy-need_upper)/local),-1)
    return features.clamp(-8,8)*g['valid'][...,None]
