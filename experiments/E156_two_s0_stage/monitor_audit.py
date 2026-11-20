"""Read-only raw-layout topology diagnostics, never used in the objective."""
import numpy as np
import torch
from .batches import decode
from .simple_loss import contact_cost
from .group_tree import minimum_group_tree
from .loss import layer_loss


def blocked(p,tree,same):
    center=p[:,:2]+p[:,2:]/2;lo=p[:,:2];hi=lo+p[:,2:]
    count=0
    for i,j in np.argwhere(tree):
        foreign=~same[i].copy();foreign[i]=False;foreign[j]=False
        a=center[i];delta=center[j]-a
        safe=np.where(np.abs(delta)>1e-12,delta,1.)
        t1=(lo-a)/safe;t2=(hi-a)/safe
        lower=np.minimum(t1,t2);upper=np.maximum(t1,t2)
        for axis in (0,1):
            if abs(delta[axis])<=1e-12:
                inside=(a[axis]>lo[:,axis]+1e-8)&(a[axis]<hi[:,axis]-1e-8)
                lower[:,axis]=np.where(inside,-np.inf,np.inf)
                upper[:,axis]=np.where(inside,np.inf,-np.inf)
        entry=np.maximum(lower.max(1),0.);leave=np.minimum(upper.min(1),1.)
        count+=int(np.any(foreign&(leave-entry>1e-7)))
    return count


def reversals(a,b):
    n=len(a);flip=np.zeros((n,n),bool)
    for axis in (0,1):
        before=a[:,None,axis]+a[:,None,axis+2]<a[None,:,axis]-1e-7
        after=b[:,None,axis]+b[:,None,axis+2]<b[None,:,axis]-1e-7
        flip|=(before&after.T)|(before.T&after)
    return int(np.triu(flip,1).sum())


@torch.no_grad()
def audit(states,g,values):
    result={}
    initial=layer_loss(states[0],g)
    for key in ('total','reference_weighted','grouping_weighted','boundary_weighted','mib_weighted','group_bottleneck_weighted'):
        result['L0_'+key]=initial[key]
    local=g['area'].clamp_min(1e-12).sqrt();scale=(local[:,:,None]+local[:,None,:])/2
    valid=g['valid'].cpu().numpy();same=(g['group']>0).cpu().numpy()
    positions={}
    for layer in (0,2,4):
        p=decode(states[layer],g)
        costs=contact_cost(p,scale)
        _,tree=minimum_group_tree(costs,g['group'],g['valid'],return_tree=True)
        coordinates=p.cpu().numpy();trees=tree.cpu().numpy()
        positions[layer]=[coordinates[i,:int(valid[i].sum())] for i in range(len(valid))]
        counts=[blocked(positions[layer][i],trees[i,:len(positions[layer][i]),:len(positions[layer][i])],
            same[i,:len(positions[layer][i]),:len(positions[layer][i])]) for i in range(len(valid))]
        result[f'L{layer}_blocked_tree_edges']=torch.tensor(counts,device=states.device,dtype=states.dtype)
        if layer:
            counts=[reversals(a,b) for a,b in zip(positions[0],positions[layer])]
            result[f'L{layer}_side_reversals']=torch.tensor(counts,device=states.device,dtype=states.dtype)
    result['L4_cases_with_side_reversal']=(result['L4_side_reversals']>0).to(states.dtype)
    result['L4_beneficial_side_reversal']=((result['L4_side_reversals']>0)&
        (values['L4_group_bottleneck_weighted']<initial['group_bottleneck_weighted']-1e-6)&
        (values['L4_overlap_total']<=values['L0_overlap_total']+1e-5)).to(states.dtype)
    return result
