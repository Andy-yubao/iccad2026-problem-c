"""Expose ten terms without replacing their original definitions."""
from .simple_loss import components
from .loss_config import A, B, COEFFICIENTS

def base_per_case(state, graph, phase='baseline'):
    out = components(state, graph, {'overlap': A, 'task': B})
    out['legacy_total'] = out['total']
    out['total'] = sum(COEFFICIENTS[key] * out[key] for key in COEFFICIENTS)
    return out

def per_case(states, graph, phase='baseline'):
    if states.ndim != 4 or states.shape[0] != 8:
        raise ValueError('Expected the 8 real block readouts, excluding the initial head')
    layers = [base_per_case(state, graph, phase) for state in states.unbind(0)]
    result = {key: .5 * layers[-1][key] + sum(layer[key] for layer in layers[:-1]) / 14.
              for key in layers[-1]}
    result['final_total'] = layers[-1]['total']
    for i, layer in enumerate(layers, 1):
        result[f'layer{i:02d}_total'] = layer['total']
    return result

def losses(states, graph, phase='baseline'):
    return {key: value.mean() for key, value in per_case(states, graph, phase).items()}
