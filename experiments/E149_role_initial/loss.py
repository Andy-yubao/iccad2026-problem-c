"""Public-hard-priority v1. Fixed objective, all modules remain predicted.

The registered old objective is preserved as a diagnostic. New public-only
penalties are independently normalized by constrained subsets and independent
of reference_scale. Raw prediction coordinates/log dimensions retain gradients
outside historical decode clamps; no geometry is overwritten.
"""
from dataclasses import asdict,dataclass
import torch
import torch.nn.functional as F
from .simple_loss import components

A=.08975162861857286
B=.3336832108335229
PUBLIC_HARD=('area','specs','constraints','valid')

@dataclass(frozen=True)
class Objective:
    reference_scale:float=1.
    public_shape_weight:float=1.
    public_preplaced_position_weight:float=1.
    grouping_multiplier:float=1.
    boundary_multiplier:float=1.

DEFAULT=Objective()

def masked_channels_mean(value,mask):
    return (value*mask[...,None]).sum((1,2))/(mask.sum(-1)*value.shape[-1]).clamp_min(1)

def public_hard_components(state,g):
    if set(g)!=set(PUBLIC_HARD):raise ValueError('Public hard objective accepts area/specs/constraints/valid only')
    valid=g['valid'];c=g['constraints'];spec=g['specs']
    fixed=valid&((c[...,0]!=0)|(c[...,1]!=0))
    pre=valid&(c[...,1]!=0)
    area=g['area'].clamp_min(1e-12)
    # No exp or clamp on the raw predicted log ratio, so extreme raw outputs
    # still receive a restoring gradient. Both width and height are supervised.
    log_area=area.log()
    predicted_log_wh=torch.stack(((log_area+state[...,2])/2,
                                  (log_area-state[...,2])/2),-1)
    safe_wh=torch.where(fixed[...,None],spec[...,2:],torch.ones_like(spec[...,2:])).clamp_min(1e-12)
    target_log_wh=safe_wh.log()
    dimension=F.smooth_l1_loss(predicted_log_wh,target_log_wh,reduction='none',beta=.1)
    position_wh=torch.where(pre[...,None],spec[...,2:],torch.ones_like(spec[...,2:])).clamp_min(1e-12)
    scale=position_wh.prod(-1).sqrt()
    target_center=torch.where(pre[...,None],spec[...,:2]+spec[...,2:]/2,torch.zeros_like(spec[...,:2]))
    position=F.smooth_l1_loss(state[...,:2]/scale[...,None],
        target_center/scale[...,None],reduction='none',beta=.1)
    return dict(public_shape=masked_channels_mean(dimension,fixed),
        public_preplaced_position=masked_channels_mean(position,pre),
        public_shape_count=fixed.sum(-1).to(state.dtype),
        public_preplaced_count=pre.sum(-1).to(state.dtype))

def base_per_case(state,g,objective=DEFAULT):
    out=components(state,g,{'overlap':A,'task':B})
    out['original_total']=out['total']
    out.update(public_hard_components(state,{k:g[k] for k in PUBLIC_HARD}))
    out['reference_weighted']=objective.reference_scale*out['reference']
    out['overlap_weighted']=A*out['overlap_total']
    out['wire_compact_weighted']=.5*B*(out['wirelength']+out['compactness'])
    out['grouping_weighted']=2*B*objective.grouping_multiplier*out['grouping_contribution']
    out['boundary_weighted']=2*B*objective.boundary_multiplier*out['boundary_contribution']
    out['public_shape_weighted']=objective.public_shape_weight*out['public_shape']
    out['public_preplaced_position_weighted']=objective.public_preplaced_position_weight*out['public_preplaced_position']
    out['total']=sum(out[k] for k in ('reference_weighted','overlap_weighted','wire_compact_weighted',
        'grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted'))
    return out

TOTAL_COMPONENTS = ('reference_weighted', 'overlap_weighted', 'wire_compact_weighted',
                    'grouping_weighted', 'boundary_weighted', 'public_shape_weighted',
                    'public_preplaced_position_weighted')


def per_case(states,g,phase,objective=DEFAULT):
    from .loss_config import ARMS
    if phase not in ARMS or states.shape[0]!=1:raise ValueError('One initial readout only')
    out=base_per_case(states[0],g,objective)
    out['full_total']=out['total'];out['full_overlap_weighted']=out['overlap_weighted']
    out['overlap_weighted']=torch.zeros_like(out['overlap_weighted'])
    out['total']=sum(out[k] for k in TOTAL_COMPONENTS if k!='overlap_weighted')
    out['final_total']=out['total'];out['original_final_total']=out['original_total']
    return out
