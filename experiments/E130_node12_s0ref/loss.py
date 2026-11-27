"""S0 reference only, L4 original, L12 reference x.8; fixed .1/.1/.8."""
import torch.nn.functional as F
from .simple_loss import components, average
from .loss_config import A, B, COEFFICIENTS, layer_weights

def reference_only(state, graph):
    mask=graph['mask']*graph['valid'][...,None]
    local=graph['area'].clamp_min(1e-12).sqrt()
    position=average(F.smooth_l1_loss(state[...,:2]/local[...,None],graph['target'][...,:2]/local[...,None],
        reduction='none',beta=.1),mask[...,:2],(1,2))
    shape=average((state[...,2]-graph['target'][...,2]).square(),mask[...,2],(1,))
    return dict(position=position,shape=shape,reference=position+.25*shape)

def base_per_case(state, graph, reference_scale=1.):
    out=components(state,graph,{'overlap':A,'task':B})
    out['original_total']=sum(COEFFICIENTS[k]*out[k] for k in COEFFICIENTS)
    out['total']=sum(COEFFICIENTS[k]*out[k]*(reference_scale if k in ('position','shape') else 1.)
        for k in COEFFICIENTS)
    out['weighted_position']=reference_scale*out['position']
    out['weighted_shape']=reference_scale*.25*out['shape']
    out['weighted_reference']=out['weighted_position']+out['weighted_shape']
    out['weighted_overlap']=A*out['overlap_total']
    out['weighted_task']=B*out['task']
    return out

def per_case(states, graph, phase='node12'):
    if states.ndim!=4 or states.shape[0]!=13:
        raise ValueError('Expected S0 followed by S1..S12; geometry depth stays12')
    weights=layer_weights(phase)
    initial=reference_only(states[0],graph)
    values={4:base_per_case(states[4],graph,1.),12:base_per_case(states[12],graph,.8)}
    result={k:sum(weights[i]*values[i][k] for i in (4,12)) for k in values[12]}
    for key in ('position','shape','reference','supervision'):
        result[key]=result[key]+weights[0]*initial['reference' if key=='supervision' else key]
    result['original_total']=result['original_total']+weights[0]*initial['reference']
    result['total']=weights[0]*initial['reference']+weights[4]*values[4]['total']+weights[12]*values[12]['total']
    result['weighted_position']=result['weighted_position']+weights[0]*initial['position']
    result['weighted_shape']=result['weighted_shape']+weights[0]*.25*initial['shape']
    result['weighted_reference']=result['weighted_reference']+weights[0]*initial['reference']
    result['initial_reference']=initial['reference']
    result['initial_position']=initial['position']; result['initial_shape']=initial['shape']
    result['layer00_total']=initial['reference']
    result['final_total']=values[12]['original_total']
    result['final_own_total']=values[12]['total']
    for i in values:
        result[f'layer{i:02d}_total']=values[i]['total']
        result[f'layer{i:02d}_original_total']=values[i]['original_total']
    for key in COEFFICIENTS: result['final_'+key]=values[12][key]
    return result
