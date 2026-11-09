"""Reference directions are supervision only, never model inputs."""
import torch
from .batches import decode

def details(state,g):
    p=decode(state,g); r=g['reference']; valid=g['valid']; n=state.shape[1]
    pc=p[...,:2]+p[...,2:]/2; rc=r[...,:2]+r[...,2:]/2
    pd=pc[:,:,None]-pc[:,None,:]; rd=rc[:,:,None]-rc[:,None,:]
    local=g['area'].clamp_min(1e-12).sqrt()
    scale=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-8)
    rext=(r[:,:,None,2:]+r[:,None,:,2:])/2
    rm=rd.abs()-rext
    pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
    mutable=g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:]
    near=(rm.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)&pairs&mutable
    use_x=rm[...,0]>=rm[...,1]
    axis=(~use_x).long()[...,None]
    # Ignore reference intersections introduced by area adaptation.
    eligible=near&(rm.amax(-1)>=-1e-6)
    direction=torch.where(use_x,rd[...,0],rd[...,1]).sign()
    # Deterministic gather backward synchronizes on the host in PyTorch2.7.1.
    # Two-axis where preserves first-axis ties and works during CUDA capture.
    signed=torch.where(use_x,pd[...,0],pd[...,1])*direction
    ext=(p[:,:,None,2:]+p[:,None,:,2:])/2
    desired=torch.where(use_x,ext[...,0],ext[...,1])
    deficit=(desired-signed).clamp_min(0)/scale
    term=(deficit.square()*eligible).sum((1,2))/valid.sum(-1).clamp_min(1)
    return term, {'eligible':eligible,'axis':axis,'direction':direction,'signed':signed,'deficit':deficit,'pd':pd,'extent':ext,'scale':scale}
