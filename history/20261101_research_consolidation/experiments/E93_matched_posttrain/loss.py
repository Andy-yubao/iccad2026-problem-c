"""One fixed objective within each phase; no dynamic/ramped coefficients."""
from .simple_loss import components
from .loss_config import COEFFICIENTS

def per_case(state, graph, phase='C'):
    return components(state, graph, COEFFICIENTS[phase])

def losses(state, graph, phase='C'):
    return {key: value.mean() for key, value in per_case(state, graph, phase).items()}
