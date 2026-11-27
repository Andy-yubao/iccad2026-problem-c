"""Reference-topology supervision with large-layout and tail risk weighting."""
import torch
import torch.nn.functional as F
from .losses import components,all_components,WEIGHTS
from .model import decode

def weights(arm,epoch,start_epoch=None):
    return {**WEIGHTS,'topology':.25 if arm=='relation_risk' else 0.,
            'tail_fraction':.25 if arm=='relation_risk' else 0.,'tail_weight':.15 if arm=='relation_risk' else 0.}

def topology_loss(state,g):
    p=decode(state,g);r=g['reference'];pc=p[...,:2]+p[...,2:]/2;rc=r[...,:2]+r[...,2:]/2
    delta=rc[:,:,None]-rc[:,None,:];extent=(r[:,:,None,2:]+r[:,None,:,2:])/2
    clear=delta.abs()-extent;axis=(clear/extent.clamp_min(1e-6)).argmax(-1)
    sign=delta.gather(-1,axis[...,None]).squeeze(-1).sign()
    pred_delta=(pc[:,:,None]-pc[:,None,:]).gather(-1,axis[...,None]).squeeze(-1)*sign
    pred_extent=((p[:,:,None,2:]+p[:,None,:,2:])/2).gather(-1,axis[...,None]).squeeze(-1)
    scale=(g['area'].sqrt()[:,:,None]+g['area'].sqrt()[:,None,:])/2
    selected=clear.gather(-1,axis[...,None]).squeeze(-1)
    near=clear.clamp_min(0).square().sum(-1).sqrt()<.25*scale
    valid=g['valid'][:,:,None]&g['valid'][:,None,:]
    valid=valid&~torch.eye(state.shape[1],device=state.device,dtype=torch.bool)[None]
    mutable=g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:]
    mask=valid&mutable&near&(selected>=-1e-6)
    target=selected.clamp_min(0).minimum(.02*scale)
    gap=(target-(pred_delta-pred_extent))/scale.clamp_min(1e-6)
    penalty=(.05*F.softplus(gap/.05)).square()
    return (penalty*mask).sum((1,2))/g['valid'].sum(1).clamp_min(1)

def losses(state,graph,targets,arm,epoch,*,include_quality=True,start_epoch=None):
    values=all_components(state,graph) if include_quality else components(state,graph)
    # Warm-start experiments use all geometry terms at fixed weights from epoch 0.
    per_case=sum(WEIGHTS[key]*values[key] for key in WEIGHTS)
    values['topology']=topology_loss(state,graph)
    n=graph['valid'].sum(1).to(per_case)
    normalizer=torch.exp((torch.arange(21,121,device=n.device,dtype=n.dtype)-120)/12).mean()
    official_importance=torch.exp((n-120)/12)/normalizer
    if include_quality:values['repair_weighted']=official_importance*values['repair']
    if arm=='relation_risk':
        per_case=per_case+.25*values['topology']
        # Frozen v5 is approximately uniform across n=21..120. The constant
        # is the exact mean exp((n-120)/12) on this prespecified support.
        importance=.25+.75*official_importance
        weighted=importance*per_case
        total=weighted.mean()+.15*per_case.topk(max(1,len(per_case)//4)).values.mean()
    elif arm=='geometry_control':total=per_case.mean()
    else:raise ValueError(arm)
    # The same predeclared module-weighted E17 repair proxy selects both arms.
    return {'total':total,**{key:value.mean() for key,value in values.items()},
            'supervision':values['local'].mean(),'quality_proxy':values['geometry'].mean()}
