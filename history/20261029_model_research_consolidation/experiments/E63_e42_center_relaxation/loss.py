"""Only center supervision changes; geometry and its aggregation stay fixed."""
import torch
import torch.nn.functional as F
from .losses import components, WEIGHTS, average
from .loss_config import configuration


def weights(arm, epoch, start_epoch=None):
    return configuration(arm)['weights']


def softened_center(error):
    shifted = error.sign() * (error.abs()-.05).clamp_min(0)
    return .5*F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none', beta=.1) + \
           .5*F.smooth_l1_loss(shifted, torch.zeros_like(shifted), reduction='none', beta=.1)


def per_case(state, graph, arm):
    config = configuration(arm)
    values = components(state, graph)
    center = values['center_local']
    if arm == 'softband005':
        local = graph['area'].clamp_min(1e-12).sqrt()[..., None]
        error = state[..., :2]/local - graph['target'][..., :2]/local
        center = average(softened_center(error), graph['mask'][..., :2], (1, 2))
    values['common_original_G'] = values['geometry']
    values['center_objective'] = config['weights']['center_local']*center
    values['total'] = values['center_objective'] + sum(
        config['weights'][name]*values[name] for name in WEIGHTS if name != 'center_local')
    values['supervision'] = values['center_objective']+.25*values['shape']+.25*values['relative_local']
    return values


def losses(state, graph, targets, arm, epoch=0, *, include_quality=True, start_epoch=None):
    return {key: value.mean() for key, value in per_case(state, graph, arm).items()}
