"""Full public task at each stage exit; S2 adds tolerant collision-aware drift."""
import torch
import torch.nn.functional as F
from .base_loss import base_per_case,A,B
from .batches import decode
from .loss_config import WEIGHTS

def mib_loss(state,g):
    ratio=state[...,2];valid=g['valid'];same=(g['mib']>0)&valid[:,:,None]&valid[:,None,:]
    diff=(ratio[:,:,None]-ratio[:,None,:]).square();peers=same.sum(-1)
    node=(diff*same).sum(-1)/peers.clamp_min(1)
    return .25*(node*(peers>0)).sum(-1)/(peers>0).sum(-1).clamp_min(1)

def layer_loss(state,g):
    row=base_per_case(state,g);row['original_full_total']=row['total']
    row['reference_weighted']=torch.zeros_like(row['reference_weighted'])
    row['mib_weighted']=2*B*mib_loss(state,g)
    keys=('overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted','mib_weighted')
    row['total']=sum(row[k] for k in keys)
    row['drift_weighted']=torch.zeros_like(row['total'])
    return row

def incoming_overlap(state,g):
    p=decode(state,g);lo=p[...,:2];hi=lo+p[...,2:]
    intersection=(torch.minimum(hi[:,:,None],hi[:,None,:])-torch.maximum(lo[:,:,None],lo[:,None,:])).clamp_min(0).prod(-1)
    n=p.shape[1];ids=torch.arange(n,device=p.device);valid=g['valid']
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
    return (intersection*pairs).sum(-1)/g['area'].clamp_min(1e-12)

def drift_terms(incoming,outgoing,g):
    incoming=incoming.detach();valid=g['valid'];c=g['constraints']
    center=(torch.linalg.vector_norm(outgoing[...,:2]-incoming[...,:2],dim=-1)/g['area'].clamp_min(1e-12).sqrt()-.05).clamp_min(0)
    ratio=((outgoing[...,2]-incoming[...,2]).abs()-.10).clamp_min(0)
    weights=torch.where(incoming_overlap(incoming,g)>.01,.25,1.)
    def reduce(value,mask):
        penalty=F.smooth_l1_loss(value,torch.zeros_like(value),beta=.1,reduction='none')
        return (penalty*weights*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    return reduce(center,valid&(c[...,1]==0)),reduce(ratio,valid&~c[...,:2].bool().any(-1))

def per_case(states,g,phase):
    if phase not in WEIGHTS or states.shape[0]!=(9 if phase=='e157_s1' else 13):raise ValueError('Unexpected stage/readouts')
    rows={k:layer_loss(states[k],g) for k in WEIGHTS[phase]}
    if phase=='e157_s2':
        center,ratio=drift_terms(states[8],states[12],g)
        rows[12]['drift_weighted']=.2*(center+.25*ratio)
        rows[12]['total']=rows[12]['total']+rows[12]['drift_weighted']
    out={key:sum(WEIGHTS[phase][k]*rows[k][key] for k in WEIGHTS[phase]) for key in rows[min(rows)]}
    terminal=max(rows);out['final_total']=rows[terminal]['total']
    out['final_drift_weighted']=rows[terminal]['drift_weighted']
    for k,row in rows.items():
        for key in ('total','overlap_weighted','overlap_total','grouping_weighted','boundary_weighted','mib_weighted','drift_weighted'):
            out[f'L{k}_{key}']=row[key]
    return out
