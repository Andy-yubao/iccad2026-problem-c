"""Selected MST corridors only; O(batch*N*N), no inference geometry feedback."""
import torch
import torch.nn.functional as F


class GatherEndpoints(torch.autograd.Function):
    """Select exactly the original endpoints, with a capture-safe gradient sum."""

    @staticmethod
    def forward(ctx, boxes, indices):
        ctx.save_for_backward(indices)
        ctx.node_count = boxes.shape[1]
        return boxes.gather(1, indices[..., None].expand(-1, -1, boxes.shape[-1]))

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        nodes = torch.arange(ctx.node_count, device=indices.device)
        assignment = (nodes[None, :, None] == indices[:, None, :]).to(grad.dtype)
        return assignment.bmm(grad), None


def components(p,tree,g,*,return_blocked=False):
    valid=g['valid'];n=valid.shape[1];area=g['area'].clamp_min(1e-12)
    scale=area.sqrt()
    # Fixed-capacity list: all forest edges, followed by masked padding.
    indices=tree.flatten(1).to(torch.int32).argsort(dim=-1,descending=True,stable=True)[:,:max(n-1,1)]
    selected=tree.flatten(1).gather(1,indices);ia=indices//n;ib=indices%n
    center=p[...,:2]+p[...,2:]/2
    pre=(g['constraints'][...,1]!=0)&valid
    locked=((g['constraints'][...,0]!=0)|(g['constraints'][...,1]!=0))&valid
    center=torch.where(pre[...,None],center.detach(),center)
    wh=torch.where(locked[...,None],p[...,2:].detach(),p[...,2:])
    lo=center-wh/2;hi=center+wh/2
    endpoints=GatherEndpoints.apply(torch.cat((lo,hi),-1),torch.cat((ia,ib),-1))
    a,b=endpoints.split(indices.shape[1],dim=1)
    al,ah=a.split(2,dim=-1)
    bl,bh=b.split(2,dim=-1)
    aw,bw=ah-al,bh-bl
    s=(scale.gather(1,ia)+scale.gather(1,ib))/2
    tangent=torch.minimum(ah,bh)-torch.maximum(al,bl)
    minwh=.01*torch.minimum(aw,bw)
    gaps=torch.stack((bl[...,0]-ah[...,0],al[...,0]-bh[...,0],bl[...,1]-ah[...,1],al[...,1]-bh[...,1]),-1)
    lack=(minwh-tangent).clamp_min(0)
    costs=gaps.square()+torch.stack((lack[...,1],lack[...,1],lack[...,0],lack[...,0]),-1).square()
    direction=costs.detach().argmin(-1);horizontal=direction<2;reverse=(direction%2)==1
    pick=lambda x:torch.where(horizontal,x[...,1],x[...,0])
    normal=lambda x:torch.where(horizontal,x[...,0],x[...,1])
    u=pick(torch.maximum(al,bl));v=pick(torch.minimum(ah,bh));length=(v-u).clamp_min(1e-12)
    normal_lo=torch.where(reverse,normal(bh),normal(ah))
    normal_hi=torch.where(reverse,normal(al),normal(bl))
    required=pick(minwh).clamp_min(1e-12)
    corridor=selected&(normal_hi>normal_lo)&((v-u)>=required)
    xl=torch.where(horizontal[...,None],lo[:,None,:,1],lo[:,None,:,0])
    xh=torch.where(horizontal[...,None],hi[:,None,:,1],hi[:,None,:,0])
    nl=torch.where(horizontal[...,None],lo[:,None,:,0],lo[:,None,:,1])
    nh=torch.where(horizontal[...,None],hi[:,None,:,0],hi[:,None,:,1])
    same=g['group'].gather(1,ia[...,None].expand(-1,-1,n))>0
    ids=torch.arange(n,device=p.device)
    active=(corridor[...,None]&valid[:,None,:]&~same&(ids!=ia[...,None])&(ids!=ib[...,None])
        &(nh>normal_lo[...,None])&(nl<normal_hi[...,None])&(xh>u[...,None])&(xl<v[...,None]))
    left=torch.maximum(xl,u[...,None]);right=torch.minimum(xh,v[...,None])
    # The interval union controls eligibility/weights, not the derivative path.
    with torch.no_grad():
        ls=torch.where(active,left,v[...,None]);rs=torch.where(active,right,u[...,None])
        order=ls.argsort(dim=-1,stable=True);ls=ls.gather(-1,order);rs=rs.gather(-1,order)
        maxima=rs.cummax(-1).values
        previous=torch.cat((u[...,None],maxima[...,:-1]),-1)
        free=torch.maximum((ls-previous).clamp_min(0).amax(-1),(v-maxima[...,-1]).clamp_min(0))
        eta=((required-free)/required).clamp(0,1)*corridor
        covered=((rs-torch.maximum(ls,previous)).clamp_min(0)).sum(-1)/length
        fraction=((right-left).clamp_min(0)/length[...,None])*active
    if return_blocked:
        return ia,ib,eta>0
    minus=xh-u[...,None]+1e-6*s[...,None]
    plus=v[...,None]-xl+1e-6*s[...,None]
    travel=torch.where(minus.detach()<=plus.detach(),minus,plus).clamp_min(0)/s.clamp_min(1e-6)[...,None]
    burden=F.smooth_l1_loss(travel,torch.zeros_like(travel),reduction='none',beta=.1)*fraction
    worst=burden.detach().argmax(-1)
    # Preserve max(dim)'s first-winner gradient without scatter backward.
    edge=eta*(burden*(ids==worst[...,None])).sum(-1)
    denominator=selected.sum(-1).clamp_min(1)
    block=edge.sum(-1)/denominator
    return dict(group_block=block,group_block_coverage=(covered*corridor).sum(-1)/denominator,
        group_block_edges=(eta>0).sum(-1).to(p.dtype))
