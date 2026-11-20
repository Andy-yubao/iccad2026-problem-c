"""Public-only swept-strip obstruction; original contact squares stay intact.

Each axis translation checks only the newly swept strip, excluding the source
footprint. Two orthogonal routes are allowed for grouping. This is a local
translation surrogate, not a certificate of global reachability.
"""
import torch


def strip_cost(source, delta, axis, obstacles, scale, allowed):
    lo=source[...,:2];hi=lo+source[...,2:]
    start=torch.where(delta.detach()>=0,hi[...,axis],lo[...,axis]+delta)
    end=torch.where(delta.detach()>=0,hi[...,axis]+delta,lo[...,axis])
    other=1-axis
    travel=(torch.minimum(end[:,:,None],obstacles[:,None,:,axis]+obstacles[:,None,:,axis+2])-
        torch.maximum(start[:,:,None],obstacles[:,None,:,axis])).clamp_min(0)
    transverse=(torch.minimum(hi[:,:,None,other],obstacles[:,None,:,other]+obstacles[:,None,:,other+2])-
        torch.maximum(lo[:,:,None,other],obstacles[:,None,:,other])).clamp_min(0)
    return ((travel*transverse)/scale.clamp_min(1e-6).square()*allowed).sum(-1)


def shift(source,dx,dy):
    return torch.stack((source[...,0]+dx,source[...,1]+dy,source[...,2],source[...,3]),-1)


def contact_translation(source,target):
    lo=source[...,:2];hi=lo+source[...,2:];tl=target[...,:2];th=tl+target[...,2:]
    dx0=tl[...,0]-hi[...,0];dx1=th[...,0]-lo[...,0]
    dy0=tl[...,1]-hi[...,1];dy1=th[...,1]-lo[...,1]
    dx=torch.where(dx0.detach().abs()<=dx1.detach().abs(),dx0,dx1)
    dy=torch.where(dy0.detach().abs()<=dy1.detach().abs(),dy0,dy1)
    mx=.01*torch.minimum(source[...,2],target[...,2])
    my=.01*torch.minimum(source[...,3],target[...,3])
    align_x=(tl[...,0]+mx-hi[...,0]).clamp_min(0)-(lo[...,0]-(th[...,0]-mx)).clamp_min(0)
    align_y=(tl[...,1]+my-hi[...,1]).clamp_min(0)-(lo[...,1]-(th[...,1]-my)).clamp_min(0)
    # Same preferred orientation as the existing contact cost; no fourth powers.
    use_x=(dx.square()+align_y.square()).detach()<=(dy.square()+align_x.square()).detach()
    return torch.where(use_x,dx,align_x),torch.where(use_x,align_y,dy)


def route_cost(source,target,obstacles,scale,allowed):
    dx,dy=contact_translation(source,target)
    xy=strip_cost(source,dx,0,obstacles,scale,allowed)+strip_cost(shift(source,dx,torch.zeros_like(dy)),dy,1,obstacles,scale,allowed)
    yx=strip_cost(source,dy,1,obstacles,scale,allowed)+strip_cost(shift(source,torch.zeros_like(dx),dy),dx,0,obstacles,scale,allowed)
    return torch.minimum(xy,yx)


def components(p,area,valid,relations,tree,required,bblo,bbhi):
    local=area.clamp_min(1e-12).sqrt();n=p.shape[1]
    scale=(local[:,:,None]+local[:,None,:])/2
    index=torch.arange(n,device=p.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(index[:,None]!=index[None,:])[None]
    # One target per non-root Prim vertex, selected only from public same-group
    # relations. bmm avoids deterministic gather-backward host synchronization.
    targets=torch.bmm(tree.to(p.dtype),p)
    edge=tree.any(-1)
    target_local=torch.bmm(tree.to(p.dtype),local[...,None]).squeeze(-1)
    obstacle_mask=pairs&~(relations>0)&edge[...,None]
    forward=route_cost(p,targets,p,scale,obstacle_mask)
    reverse_scale=(target_local[:,:,None]+local[:,None,:])/2
    reverse=route_cost(targets,p,p,reverse_scale,obstacle_mask)
    group_sum=(.5*(forward+reverse)*edge).sum(-1)
    lo=p[...,:2];hi=lo+p[...,2:]
    distances=(bblo[:,None,0]-lo[...,0],bbhi[:,None,0]-hi[...,0],
        bbhi[:,None,1]-hi[...,1],bblo[:,None,1]-lo[...,1])
    blocked=torch.stack([strip_cost(p,d,a,p,scale,pairs) for d,a in zip(distances,(0,0,1,1))],-1)
    boundary_sum=((blocked*required).sum(-1)/required.sum(-1).clamp_min(1)).sum(-1)
    ng=edge.sum(-1);nb=required.any(-1).sum(-1);den=(ng+nb).clamp_min(1)
    return dict(grouping_blockage=group_sum/ng.clamp_min(1),boundary_blockage=boundary_sum/nb.clamp_min(1),
        grouping_blockage_contribution=group_sum/den,boundary_blockage_contribution=boundary_sum/den)
