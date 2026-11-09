"""Each arm has one immutable objective for backward, monitoring and selection."""
from .arms import ARMS
from .losses import WEIGHTS


def configuration(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    weights = dict(WEIGHTS)
    if arm == 'weight075':
        weights['center_local'] = .75
    return {'revision': 'E63.fixed_center_relaxation.v1', 'weights': weights,
            'center': ('0.5*rho(e)+0.5*rho(sign(e)*relu(abs(e)-0.05)); e=state_xy/sqrt(area)-target_xy/sqrt(area); rho=SmoothL1(beta=.1)'
                       if arm == 'softband005' else 'original E42 axiswise local SmoothL1(beta=.1)'),
            'softband_tau': .05 if arm == 'softband005' else None,
            'softband_original_fraction': .5 if arm == 'softband005' else None,
            'count_factor': False, 'aggregation': 'original E42 per-layout terms; equal layout mean',
            'schedule': 'full fixed weights from the start of post-training; no ramp',
            'selection': 'dev_monitor500 arithmetic mean of this same total',
            'historical_difference': 'E42 originally ramped five terms in its first two scratch epochs and selected with G+occupancy; this post-training follows current fixed-Loss protocol using G only',
            'auxiliary_original_G': 'interpretation only; never selects, schedules or stops',
            'extra_occupancy_objectives': False}
