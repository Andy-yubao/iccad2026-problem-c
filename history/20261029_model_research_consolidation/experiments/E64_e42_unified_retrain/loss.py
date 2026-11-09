"""Frozen E42 eight-term G; one function for backward and all monitoring."""
from .losses import components, WEIGHTS
from .loss_config import configuration


def weights(arm, epoch=0, start_epoch=None):
    return configuration(arm)['weights']


def per_case(state, graph, arm):
    configuration(arm)
    values = components(state, graph)
    values['total'] = values['geometry']
    values['supervision'] = values['local']
    return values


def losses(state, graph, targets, arm, epoch=0, *, include_quality=True, start_epoch=None):
    return {key: value.mean() for key, value in per_case(state, graph, arm).items()}
