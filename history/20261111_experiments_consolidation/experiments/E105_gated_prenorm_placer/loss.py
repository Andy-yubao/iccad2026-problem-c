"""Expose ten terms without replacing their original definitions."""
from .simple_loss import components
from .loss_config import A, B, COEFFICIENTS

def per_case(state, graph, phase='baseline'):
    out = components(state, graph, {'overlap': A, 'task': B})
    out['legacy_total'] = out['total']
    out['total'] = sum(COEFFICIENTS[key] * out[key] for key in COEFFICIENTS)
    return out

def losses(state, graph, phase='baseline'):
    return {key: value.mean() for key, value in per_case(state, graph, phase).items()}
