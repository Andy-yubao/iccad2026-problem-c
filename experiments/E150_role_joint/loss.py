"""Identical pointwise base Loss; additional directly supervised initial readout."""
import torch
from .base_loss import base_per_case,TOTAL_COMPONENTS,DEFAULT
from .loss_config import ARMS,WEIGHTS,NO_OVERLAP

def per_case(states,g,phase,objective=DEFAULT):
    if phase not in ARMS or states.ndim!=4 or states.shape[0]!=13:raise ValueError('Expected actual initial plus12 refinement readouts')
    values={}
    for k in WEIGHTS:
        row=base_per_case(states[k],g,objective)
        row['full_total']=row['total'];row['full_overlap_weighted']=row['overlap_weighted']
        if k in NO_OVERLAP:
            row['overlap_weighted']=torch.zeros_like(row['overlap_weighted'])
            row['total']=sum(row[name] for name in TOTAL_COMPONENTS if name!='overlap_weighted')
        values[k]=row
    out={name:sum(WEIGHTS[k]*values[k][name] for k in WEIGHTS) for name in values[12]}
    out.update(final_total=values[12]['total'],original_final_total=values[12]['original_total'],full_total=values[12]['full_total'],
        initial_no_overlap_total=values[0]['total'],initial_full_total=values[0]['full_total'])
    for k,row in values.items():
        for name in ('total','full_total','overlap_weighted','full_overlap_weighted',*TOTAL_COMPONENTS):out[f'layer{k:02d}_{name}']=row[name]
    return out
