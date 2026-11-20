"""Apply the same original-area/no-MIB complete Loss to the unique final draft."""
from .simple_loss import components
from .loss_config import A, B, COEFFICIENTS, ARMS


def base_per_case(state, graph):
    out = components(state, graph, {'overlap': A, 'task': B})
    out['total'] = sum(COEFFICIENTS[k]*out[k] for k in COEFFICIENTS)
    return out


def per_case(state, graph, phase=None):
    if phase is not None and phase not in ARMS:
        raise ValueError(phase)
    if state.ndim != 3 or state.shape[-1] != 3:
        raise ValueError('Expected a single final draft [batch, nodes, 3]')
    out = base_per_case(state, graph)
    out['final_total'] = out['total']
    return out
