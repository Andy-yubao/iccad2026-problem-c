"""Evaluate full layout losses only at the registered supervision locations."""
from .simple_loss import components
from .loss_config import A, B, COEFFICIENTS, layer_weights

def base_per_case(state, graph):
    out = components(state, graph, {'overlap': A, 'task': B})
    out['total'] = sum(COEFFICIENTS[k]*out[k] for k in COEFFICIENTS)
    return out

def per_case(states, graph, phase='terminal'):
    if states.ndim != 4 or states.shape[0] != 12:
        raise ValueError('Expected all twelve real block readouts')
    weights = layer_weights(phase)
    values = {i: base_per_case(states[i-1], graph) for i in weights}
    result = {k: sum(weights[i]*values[i][k] for i in weights) for k in values[12]}
    result['final_total'] = values[12]['total']
    for i in values: result[f'layer{i:02d}_total'] = values[i]['total']
    return result
