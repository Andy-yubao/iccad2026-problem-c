"""Original E42 eight-term geometry Loss, fixed for backward and monitoring."""
from .losses import components, WEIGHTS
from .loss_config import configuration


def weights(arm, epoch, start_epoch=None):
    return configuration(arm)['weights']


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    configuration(arm)
    values = components(state, graph)
    total = sum(WEIGHTS[name]*values[name].mean() for name in WEIGHTS)
    return {'total': total, **{key: value.mean() for key, value in values.items()},
            'supervision': values['local'].mean()}
