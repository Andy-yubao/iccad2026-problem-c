"""This definition also binds subsequent model-capacity work unless user revises it."""
from .arms import ARMS
from .losses import WEIGHTS


def configuration(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return {'revision': 'E64.frozen_E42_G.v1', 'weights': dict(WEIGHTS),
            'center': 'original E42 axiswise local SmoothL1(beta=.1)',
            'count_factor': False,
            'aggregation': 'original E42 per-layout terms, then equal-layout arithmetic mean',
            'schedule': 'full fixed weights from first scratch batch; no ramp or late relaxation',
            'selection': 'same total, arithmetic mean on frozen v5 dev_monitor500',
            'historical_difference': 'remove E42 two-epoch ramp and G+occupancy selector; retain full-weight backward G',
            'extra_occupancy_objectives': False}
