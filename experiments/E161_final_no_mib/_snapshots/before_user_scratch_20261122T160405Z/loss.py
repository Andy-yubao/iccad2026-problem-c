"""Original-area/no-MIB objective, plus one selected-MST blockage term."""
from .base_loss import base_per_case,B,Objective
from .loss_config import ARMS,REFERENCE,OVERLAP,BLOCK
from .blockage import components


def per_case(states,g,phase):
    k=ARMS.index(phase)
    if states.shape[0]!=k+1:raise ValueError('One real exit per constructed stage required')
    row=base_per_case(states[-1],g,Objective(reference_scale=REFERENCE[k]))
    row['original_overlap_weighted']=row['overlap_weighted']
    row['overlap_weighted']=OVERLAP[k]*row['overlap_weighted']
    row.update(components(states[-1],g))
    valid=g['valid'];n=valid.shape[1]
    pairs=valid[:,:,None]&valid[:,None,:]
    ids=__import__('torch').arange(n,device=valid.device)
    peers=((g['group']>0)&pairs&(ids[:,None]!=ids[None,:])[None]).sum(-1).to(states.dtype)
    ng=(peers/(peers+1)).sum(-1)
    nb=((g['constraints'][...,4]!=0)&valid).sum(-1)
    row['group_block_weighted']=2*B*BLOCK*ng/(ng+nb).clamp_min(1)*row['group_block']
    keys=('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted',
        'boundary_weighted','public_shape_weighted','public_preplaced_position_weighted','group_block_weighted')
    row['total']=sum(row[name] for name in keys)
    return row
