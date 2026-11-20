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

def frontend_components(state,g):
    """C's weak S0 objective; raw outputs, per-case subset means, zero if empty."""
    valid=g['valid'];constraints=g['constraints']
    nonpre=valid & (constraints[...,1]==0)
    free=valid & (constraints[...,0]==0) & (constraints[...,1]==0)
    local=g['area'].clamp_min(1e-12).sqrt()
    excess=((state[...,:2]-g['target'][...,:2]).abs()/local[...,None]-.25).clamp_min(0)
    center=masked_channels_mean(F.smooth_l1_loss(excess,torch.zeros_like(excess),reduction='none',beta=.1),nonpre)
    shape=((state[...,2]-g['target'][...,2]).square()*free).sum(-1)/free.sum(-1).clamp_min(1)
    hard=public_hard_components(state,{k:g[k] for k in PUBLIC_HARD})
    return dict(center_tolerance=center,free_shape=shape,**hard,
        total=center+.05*shape+hard['public_shape']+hard['public_preplaced_position'])


def per_case(states,g,phase,objective=DEFAULT):
    from .loss_config import layer_weights
    if states.ndim!=4 or states.shape[0]!=13:raise ValueError('Expected existing S0 plus S1..S12')
    weights=layer_weights(phase)
    values={k:base_per_case(states[k],g,objective) for k in weights}
    terms=('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted',
           'boundary_weighted','public_shape_weighted','public_preplaced_position_weighted')
    out={name:sum(weights[k]*values[k][name] for k in weights) for name in terms}
    out['total']=sum(weights[k]*values[k]['total'] for k in weights)
    if phase=='e119_C':
        coarse={k:coarse_components(states[k],g) for k in (3,6,9)}
        out={name:.8*values[12][name] for name in terms}
        out['total']=.8*values[12]['total']+sum(weights[k]*coarse[k]['total'] for k in coarse)
        out['reference_weighted']=out['reference_weighted']+sum(weights[k]*(coarse[k]['center_tolerance']+.05*coarse[k]['free_shape']+.25*coarse[k]['strong_order']) for k in coarse)
        for field,name in (('public_shape','public_shape_weighted'),('public_preplaced_position','public_preplaced_position_weighted')):
            out[name]=out[name]+sum(weights[k]*coarse[k][field] for k in coarse)
        for k in coarse:out[f'layer{k:02d}_coarse']=coarse[k]['total']
    out.update(final_total=values[12]['total'],original_final_total=values[12]['original_total'])
    for k in values:out[f'layer{k:02d}_total']=values[k]['total']
    return out

def coarse_components(state,g):
    out=frontend_components(state,g)
    local=g['area'].clamp_min(1e-12).sqrt()
    scale=(local[:,:,None]+local[:,None,:])/2
    target=g['target'][...,:2]
    delta=target[:,:,None,:]-target[:,None,:,:]
    half=(g['reference'][:,:,None,2:]+g['reference'][:,None,:,2:])/2
    valid=g['valid'];pre=g['constraints'][...,1]!=0
    pairs=valid[:,:,None]&valid[:,None,:]&~(pre[:,:,None]&pre[:,None,:])
    strong=(delta.abs()-half>=.25*scale[...,None])&pairs[...,None]
    predicted=(state[:,:,:2][:,:,None,:]-state[:,:,:2][:,None,:,:])/scale[...,None].clamp_min(1e-6)
    violation=(.25-delta.sign()*predicted).clamp_min(0)
    penalty=F.smooth_l1_loss(violation,torch.zeros_like(violation),reduction='none',beta=.1)*strong
    worst=penalty.amax(2);eligible=strong.any(2)
    order=(worst*eligible).sum((1,2))/eligible.sum((1,2)).clamp_min(1)
    out.update(strong_order=order,total=out['total']+.25*order)
    return out

def objective_forward(model,batch,arm):
    """Only public fields enter either forward; labels used strictly in Loss."""
    from .batches import inputs_only
    clean=model(inputs_only(batch));out=per_case(clean,batch,arm)
    out['clean_objective']=out['total']
    if arm=='e119_A':
        return out,clean
    damaged=model(inputs_only(batch),corrupt=True)
    recovery=base_per_case(damaged[12],batch)
    for name in ('total','reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted'):
        out[name]=.8*out[name]+.2*recovery[name]
    out['repair_total']=recovery['total']
    return out,torch.stack((clean,damaged))


def configuration(objective=DEFAULT):
    return dict(version='public-hard-priority.v1',coefficients=asdict(objective),A=A,B=B,
        layer_weights={'4':.2,'12':.8},normalization='per-case means within each constrained subset; zero if subset empty',
        shape='F or P log width/height vs public immutable dimensions; Huber beta .1',
        position='P raw center vs public specified center, scaled by sqrt(spec area); Huber beta .1',
        raw_domain='public hard targets bypass decode clamps in loss only; predictions unchanged',
        all_modules_predicted=True,specification_overwrite=False,MIB=False,
        fixed_during_run=True,same_backward_monitor_best_lr_earlystop=True,
        historical_original_loss='diagnostic only',training_authorized=False)
