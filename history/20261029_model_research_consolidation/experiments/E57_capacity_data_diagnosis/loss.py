"""Each run has one fixed objective shared by backward, monitoring and evaluation."""
from .losses import components, WEIGHTS
from .loss_config import configuration


def weights(arm, epoch, start_epoch=None):
    return configuration(arm)['weights']


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    configuration(arm)
    if arm.startswith('reference'):
        from .reference_loss import losses as reference
        return reference(state, graph, targets, arm, epoch, include_quality=include_quality, start_epoch=start_epoch)
    values = components(state, graph)
    total = sum(WEIGHTS[key]*values[key].mean() for key in WEIGHTS)
    return {'total': total, **{key: value.mean() for key, value in values.items()},
            'supervision': values['local'].mean()}
