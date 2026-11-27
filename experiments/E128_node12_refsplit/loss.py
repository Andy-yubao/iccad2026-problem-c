"""Fixed per-layer reference allocation; unchanged physical primitives."""
from .simple_loss import components
from .loss_config import A, B, COEFFICIENTS, layer_weights, reference_weights

def base_per_case(state, graph, reference_scale=1.):
    out = components(state, graph, {'overlap': A, 'task': B})
    out['original_total'] = sum(COEFFICIENTS[k]*out[k] for k in COEFFICIENTS)
    out['total'] = sum(COEFFICIENTS[k]*out[k]*(reference_scale if k in ('position','shape') else 1.)
                       for k in COEFFICIENTS)
    out['weighted_position'] = reference_scale*out['position']
    out['weighted_shape'] = reference_scale*.25*out['shape']
    out['weighted_reference'] = out['weighted_position']+out['weighted_shape']
    out['weighted_overlap'] = A*out['overlap_total']
    out['weighted_task'] = B*out['task']
    return out

def per_case(states, graph, phase='node12'):
    if states.ndim != 4 or states.shape[0] != 12:
        raise ValueError('Expected all twelve real block readouts')
    weights = layer_weights(phase); refs = reference_weights(phase)
    values = {i: base_per_case(states[i-1], graph, refs[i]) for i in weights}
    result = {k: sum(weights[i]*values[i][k] for i in weights) for k in values[12]}
    result['final_total'] = values[12]['original_total']
    result['final_own_total'] = values[12]['total']
    for i in values:
        result[f'layer{i:02d}_total'] = values[i]['total']
        result[f'layer{i:02d}_original_total'] = values[i]['original_total']
    for key in COEFFICIENTS:
        result['final_'+key] = values[12][key]
    return result
