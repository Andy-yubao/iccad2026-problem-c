"""Current MIB disagreement and group-obstacle diagnostics, public and detached."""
import torch
@torch.no_grad()
def extra_features(p,g):
    valid=g['valid'];n=p.shape[1];eye=torch.eye(n,device=p.device,dtype=torch.bool)[None]
    same=(g['mib']>0)&valid[:,:,None]&valid[:,None,:]
    weight=(same|eye).to(p.dtype)*valid[:,None,:];weight=weight/weight.sum(-1,keepdim=True).clamp_min(1)
    ratio=(p[...,2]/p[...,3].clamp_min(1e-12)).log();mean=(weight@ratio[...,None]).squeeze(-1)
    variance=(weight@((ratio-mean).square()[...,None])).squeeze(-1)
    locked=valid&g['constraints'][...,:2].bool().any(-1)
    rigid=(same|eye)&locked[:,None,:];rw=rigid.to(p.dtype)/rigid.sum(-1,keepdim=True).clamp_min(1)
    known=(g['specs'][...,2].clamp_min(1e-12)/g['specs'][...,3].clamp_min(1e-12)).log()
    target=(rw@known[...,None]).squeeze(-1);has=rigid.any(-1)
    mib=torch.stack((same.any(-1),same.sum(-1).to(p.dtype).log1p(),ratio-mean,variance.sqrt(),has,(ratio-target)*has),-1)
    # Each module's group rectangle: foreign occupancy and its own relation to it.
    group=(g['group']>0)|eye;group&=valid[:,:,None]&valid[:,None,:]
    lo=p[...,:2];hi=lo+p[...,2:]
    gl=lo[:,None].masked_fill(~group[...,None],float('inf')).amin(2)
    gh=hi[:,None].masked_fill(~group[...,None],-float('inf')).amax(2)
    gl=torch.where(valid[...,None],gl,torch.zeros_like(gl));gh=torch.where(valid[...,None],gh,torch.zeros_like(gh))
    intersection=(torch.minimum(gh[:,:,None],hi[:,None])-torch.maximum(gl[:,:,None],lo[:,None])).clamp_min(0).prod(-1)
    foreign=valid[:,None,:]&~group;area=g['area'].clamp_min(1e-12)
    load=(intersection*foreign).sum(-1)/area
    count=((intersection>0)&foreign).sum(-1).to(p.dtype).log1p()
    span=(gh-gl)/area.sqrt()[...,None]
    return torch.cat((mib,load.log1p()[...,None],count[...,None],span.asinh()),-1)*valid[...,None]
