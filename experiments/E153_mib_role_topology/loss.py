"""Common MIB/public loss with case/group bottlenecks and .75 S0 reference."""
import torch
from .base_loss import base_per_case,A,B
from .simple_loss import contact_cost
from .group_tree import minimum_group_tree
from .batches import decode
from .loss_config import WEIGHTS,REFERENCE
def mib_loss(state,g):
    ratio=state[...,2];valid=g['valid'];same=(g['mib']>0)&valid[:,:,None]&valid[:,None,:]
    diff=(ratio[:,:,None]-ratio[:,None,:]).square()
    # log-width/log-height differences, equal-area task; independent outputs.
    peers=same.sum(-1);node=(diff*same).sum(-1)/peers.clamp_min(1)
    return .25*(node*(peers>0)).sum(-1)/(peers>0).sum(-1).clamp_min(1)
def grouping_bottleneck(state,g):
    p=decode(state,g);local=g['area'].clamp_min(1e-12).sqrt();scale=(local[:,:,None]+local[:,None,:])/2
    costs=contact_cost(p,scale);_,tree=minimum_group_tree(costs,g['group'],g['valid'],return_tree=True)
    # Mean tree cost is already in the base loss; add the worst selected edge per case.
    return (costs*tree).amax((1,2))
def drift(incoming,outgoing,g):
    local=g['area'].clamp_min(1e-12).sqrt();valid=g['valid'];c=g['constraints']
    center=(torch.linalg.vector_norm(outgoing[...,:2]-incoming.detach()[...,:2],dim=-1)/local-.05).clamp_min(0)
    ratio=((outgoing[...,2]-incoming.detach()[...,2]).abs()-.10).clamp_min(0)
    import torch.nn.functional as F
    def reduce(v,mask):return (F.smooth_l1_loss(v,torch.zeros_like(v),beta=.1,reduction='none')*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    return reduce(center,valid&(c[...,1]==0))+.25*reduce(ratio,valid&~c[...,:2].bool().any(-1))
def layer_loss(state,g,reference,overlap):
    row=base_per_case(state,g);row['original_full_total']=row['total']
    row['reference_weighted']=reference*row['reference_weighted']
    if not overlap:row['overlap_weighted']=torch.zeros_like(row['overlap_weighted'])
    row['mib_weighted']=2*B*mib_loss(state,g)
    row['group_bottleneck_weighted']=2*B*grouping_bottleneck(state,g)
    keys=('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted','mib_weighted','group_bottleneck_weighted')
    row['total']=sum(row[k] for k in keys)
    return row
def per_case(states,g,phase):
    if phase=='e153_initial':
        assert states.shape[0]==1
        return layer_loss(states[0],g,1.,False)
    if phase!='e153_backend' or states.shape[0]!=13:raise ValueError('Stage/readout mismatch')
    rows={k:layer_loss(states[k],g,REFERENCE[k],k!=4) for k in WEIGHTS}
    rows[12]['drift_weighted']=.2*drift(states[8],states[12],g)
    rows[12]['total']=rows[12]['total']+rows[12]['drift_weighted']
    out={name:sum(WEIGHTS[k]*rows[k][name] for k in WEIGHTS) for name in rows[4]}
    out['final_total']=rows[12]['total'];out['drift_weighted']=rows[12]['drift_weighted']
    for k,row in rows.items():
        for name in ('total','reference_weighted','overlap_weighted','grouping_weighted','mib_weighted','group_bottleneck_weighted'):out[f'L{k}_{name}']=row[name]
    return out
