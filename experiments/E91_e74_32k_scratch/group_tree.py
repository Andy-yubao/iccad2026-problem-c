"""GPU Prim forest; detached choices, gradients through chosen contact costs."""
import torch

def minimum_group_tree(cost,relations,valid):
    b,n,_=cost.shape;device=cost.device
    same=(relations>0)&valid[:,:,None]&valid[:,None,:]
    eligible=same.any(-1);visited=torch.zeros_like(eligible)
    best=torch.full((b,n),float('inf'),device=device,dtype=cost.dtype)
    parent=torch.zeros((b,n),device=device,dtype=torch.long)
    choices=cost.detach()
    index=torch.arange(n,device=device)[None,:]
    tree=torch.zeros((b,n,n),device=device,dtype=torch.bool)
    for _ in range(n):
        active=eligible&~visited
        value,vertex=best.masked_fill(~active,float('inf')).min(-1)
        connected=torch.isfinite(value);root=active.to(torch.int32).argmax(-1)
        vertex=torch.where(connected,vertex,root)
        par=parent.gather(1,vertex[:,None]).squeeze(1)
        # Keep discrete choices detached. Gathering differentiable edge costs
        # triggers a deterministic scatter backward with a host synchronization
        # in this runtime; a final masked reduction has the same selected costs.
        edge=(index[:,:,None]==vertex[:,None,None])&(index[:,None,:]==par[:,None,None])
        tree=tree|(edge&connected[:,None,None])
        visited=visited|((index==vertex[:,None])&active)
        candidate=choices.gather(1,vertex[:,None,None].expand(-1,1,n)).squeeze(1)
        allowed=same.gather(1,vertex[:,None,None].expand(-1,1,n)).squeeze(1)&~visited
        improve=allowed&(candidate<best)
        best=torch.where(improve,candidate,best);parent=torch.where(improve,vertex[:,None],parent)
    peers=same.sum(-1).to(cost.dtype)
    return (cost*tree).sum((1,2))/(peers/(peers+1)).sum(-1).clamp_min(1)
