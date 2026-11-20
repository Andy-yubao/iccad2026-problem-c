"""Two-S0 deep supervision and a terminal relative-overlap guard."""
import torch
from .base_loss import base_per_case,A,B
from .simple_loss import contact_cost
from .group_tree import minimum_group_tree
from .batches import decode
from .loss_config import WEIGHTS,REFERENCE

def mib_loss(state,g):
    ratio=state[...,2];valid=g['valid'];same=(g['mib']>0)&valid[:,:,None]&valid[:,None,:]
    diff=(ratio[:,:,None]-ratio[:,None,:]).square();peers=same.sum(-1)
    node=(diff*same).sum(-1)/peers.clamp_min(1)
    return .25*(node*(peers>0)).sum(-1)/(peers>0).sum(-1).clamp_min(1)

def grouping_bottleneck(state,g):
    p=decode(state,g);local=g['area'].clamp_min(1e-12).sqrt()
    costs=contact_cost(p,(local[:,:,None]+local[:,None,:])/2)
    _,tree=minimum_group_tree(costs,g['group'],g['valid'],return_tree=True)
    return (costs*tree).amax((1,2))

def overlap_total(state,g):
    p=decode(state,g);valid=g['valid'];n=valid.shape[1];count=valid.sum(-1).clamp_min(1)
    area=g['area'].clamp_min(1e-12);local=area.sqrt();lo=p[...,:2];hi=lo+p[...,2:]
    center=(lo+hi)/2;delta=center[:,:,None]-center[:,None,:]
    scale=(local[:,:,None]+local[:,None,:])/2
    ids=torch.arange(n,device=state.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
    upper=pairs&(ids[:,None]<ids[None,:])[None]
    slack=(p[:,:,None,2:]+p[:,None,:,2:])/2-delta.abs()
    depth=torch.minimum(slack[...,0],slack[...,1]).clamp_min(0)
    penetration=((depth/scale.clamp_min(1e-6)).square()*upper).sum((1,2))/count
    intersection=(torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(lo[:,:,None],lo[:,None,:])).clamp_min(0)
    load=(intersection[...,0]*intersection[...,1]*pairs).sum(-1)/area*valid
    return penetration+load.sum(-1)/count+(load-.10).clamp_min(0).masked_fill(~valid,0).amax(-1)

def layer_loss(state,g,reference=.75):
    row=base_per_case(state,g)
    row['original_full_total']=row['total']
    row['reference_weighted']=reference*row['reference_weighted']
    row['overlap_weighted']=torch.zeros_like(row['overlap_weighted'])
    row['mib_weighted']=2*B*mib_loss(state,g)
    row['group_bottleneck_weighted']=.2*grouping_bottleneck(state,g)
    keys=('reference_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted',
        'public_shape_weighted','public_preplaced_position_weighted','mib_weighted','group_bottleneck_weighted')
    row['total']=sum(row[k] for k in keys)
    return row

def per_case(states,g,phase):
    if phase!='e156_s0_backend' or states.shape[0]!=5:raise ValueError('Expected frozen L0 and four S0 readouts')
    rows={k:layer_loss(states[k],g,REFERENCE[k]) for k in WEIGHTS}
    baseline=overlap_total(states[0],g).detach()
    guard=A*(rows[4]['overlap_total']-baseline).clamp_min(0)
    out={name:sum(WEIGHTS[k]*rows[k][name] for k in WEIGHTS) for name in rows[2]}
    out['total']=out['total']+guard
    out['relative_overlap_weighted']=guard
    out['L0_overlap_total']=baseline
    out['final_total']=rows[4]['total']+guard
    for k,row in rows.items():
        for name in ('total','reference_weighted','grouping_weighted','boundary_weighted','mib_weighted',
                     'group_bottleneck_weighted','overlap_total'):
            out[f'L{k}_{name}']=row[name]
    return out
