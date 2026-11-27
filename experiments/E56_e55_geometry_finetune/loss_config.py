"""E42's eight backward terms; full fixed weights in every phase."""
from .losses import WEIGHTS


def configuration(arm):
    if arm != 'e42_geometry':
        raise ValueError(arm)
    return {'revision': 'E56.E42_fixed_eight_terms.v1', 'weights': dict(WEIGHTS),
            'count_factor': False, 'aggregation': 'original E42 per-layout terms; equal layout mean',
            'schedule': 'full fixed weights from the start of post-training; no ramp',
            'selection': 'dev_monitor500 arithmetic mean of this same total',
            'historical_difference': 'E42 originally ramped five terms in its first two scratch epochs and selected with G+occupancy; this post-training follows current fixed-Loss protocol using G only',
            'extra_occupancy_objectives': False}
