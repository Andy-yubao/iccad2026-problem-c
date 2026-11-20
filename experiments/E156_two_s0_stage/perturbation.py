"""Train-only public layout augmentation; no labels or immutable-input edits."""
import torch


def perturb(state,g,draw=None):
    b,n,_=state.shape;device=state.device
    u=torch.rand(b,device=device) if draw is None else draw
    jitter=u<.125;swap=(u>=.125)&(u<.25)
    local=g['area'].clamp_min(1e-12).sqrt()[...,None]
    offset=(2*torch.rand(b,n,2,device=device)-1)*.1*local
    score=torch.rand(b,n,device=device).masked_fill(~g['valid'],-1)
    first=score.argmax(-1);ids=torch.arange(n,device=device)[None,:]
    center=state[...,:2]
    start=center.gather(1,first[:,None,None].expand(-1,1,2))
    distance=(center-start).square().sum(-1)
    second=distance.masked_fill(~g['valid']|(ids==first[:,None]),float('inf')).argmin(-1)
    end=center.gather(1,second[:,None,None].expand(-1,1,2))
    exchanged=torch.where((ids==first[:,None])[...,None],end,
        torch.where((ids==second[:,None])[...,None],start,center))
    changed=torch.where(jitter[:,None,None],center+offset,center)
    changed=torch.where(swap[:,None,None],exchanged,changed)
    return torch.cat((changed,state[...,2:]),-1)*g['valid'][...,None]
