"""Original L0 objective plus one quarter of the complete original overlap group."""
from .base_loss import base_per_case,B
from .loss_config import ARMS,OVERLAP_SCALE

def mib_loss(state,g):
    ratio=state[...,2];valid=g['valid'];same=(g['mib']>0)&valid[:,:,None]&valid[:,None,:]
    diff=(ratio[:,:,None]-ratio[:,None,:]).square();peers=same.sum(-1)
    node=(diff*same).sum(-1)/peers.clamp_min(1)
    return .25*(node*(peers>0)).sum(-1)/(peers>0).sum(-1).clamp_min(1)

def layer_loss(state,g):
    row=base_per_case(state,g)
    row['original_full_total']=row['total']
    row['original_overlap_weighted']=row['overlap_weighted']
    row['overlap_weighted']=OVERLAP_SCALE*row['original_overlap_weighted']
    row['mib_weighted']=2*B*mib_loss(state,g)
    row['group_bottleneck_weighted']=row['mib_weighted']*0
    keys=('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted',
          'boundary_weighted','public_shape_weighted','public_preplaced_position_weighted','mib_weighted')
    row['total']=sum(row[k] for k in keys)
    row['original_L0_no_overlap_total']=sum(row[k] for k in keys if k!='overlap_weighted')
    row['full_overlap_diagnostic_total']=row['original_L0_no_overlap_total']+row['original_overlap_weighted']
    return row

def per_case(states,g,phase):
    if phase not in ARMS or states.shape[0]!=1:raise ValueError('Only one L0 output is permitted')
    return layer_loss(states[0],g)
