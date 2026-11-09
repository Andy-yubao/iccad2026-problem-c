"""One fixed final-layout objective plus set-valued relation supervision."""
import torch
from .batches import decode
from .simple_loss import components
from .loss_config import A,B,COEFFICIENTS,RELATION_WEIGHT


def per_case(output,graph,phase='baseline'):
    state = output['state']
    out = components(state,graph,{'overlap':A,'task':B})
    logits = output['relation_logits']
    ref = graph['reference']
    lo,hi = ref[...,:2],ref[...,:2]+ref[...,2:]
    tol = 1e-6
    allowed = torch.stack((hi[:,:,None,0]<=lo[:,None,:,0]+tol,
                           hi[:,None,:,0]<=lo[:,:,None,0]+tol,
                           hi[:,:,None,1]<=lo[:,None,:,1]+tol,
                           hi[:,None,:,1]<=lo[:,:,None,1]+tol),-1)
    valid = graph['valid']
    n = valid.shape[1]
    ids = torch.arange(n,device=state.device)
    movable = graph['mask'][...,0].bool()
    pairs = valid[:,:,None]&valid[:,None,:]&(ids[:,None]<ids[None,:])[None]
    supervised = pairs&allowed.any(-1)&(movable[:,:,None]|movable[:,None,:])
    count = supervised.sum((1,2)).clamp_min(1)
    log_mass = torch.logsumexp(logits.masked_fill(~allowed,-1e4),-1)-torch.logsumexp(logits,-1)
    out['relation'] = (-log_mass*supervised).sum((1,2))/count
    chosen = logits.argmax(-1)
    selected_valid = allowed.gather(-1,chosen[...,None]).squeeze(-1)
    out['relation_reference_accuracy'] = (selected_valid*supervised).sum((1,2))/count
    out['relation_confidence'] = (output['relation_confidence']*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)
    p = decode(state,graph)
    plo,phi = p[...,:2],p[...,:2]+p[...,2:]
    satisfied = torch.stack((phi[:,:,None,0]<=plo[:,None,:,0]+tol,
                             phi[:,None,:,0]<=plo[:,:,None,0]+tol,
                             phi[:,:,None,1]<=plo[:,None,:,1]+tol,
                             phi[:,None,:,1]<=plo[:,:,None,1]+tol),-1)
    out['relation_realized_fraction'] = (satisfied.gather(-1,chosen[...,None]).squeeze(-1)*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)
    local = graph['area'].clamp_min(1e-12).sqrt()
    motion = ((state[...,:2]-output['initial_state'][...,:2])/local[...,None]).square().sum(-1).sqrt()
    out['geometry_displacement_local'] = (motion*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    out['legacy_total'] = out['total']
    out['total'] = sum(COEFFICIENTS[key]*out[key] for key in COEFFICIENTS)+RELATION_WEIGHT*out['relation']
    return out


def losses(output,graph,phase='baseline'):
    return {key:value.mean() for key,value in per_case(output,graph,phase).items()}
