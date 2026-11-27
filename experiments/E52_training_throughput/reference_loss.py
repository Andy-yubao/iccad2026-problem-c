"""One fixed Loss for optimization, train/dev monitoring and authorized tests."""
from .reference_losses import components, WEIGHTS
from .reference_module_loss import LOSS_WEIGHT, components as module_components


def weights(arm, epoch, start_epoch=None):
    return {**WEIGHTS, 'module_overlap': LOSS_WEIGHT}


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    # Compatibility arguments never change terms/weights by split, epoch or mode.
    if arm != 'main':
        raise ValueError('E52 unified-v2 has one fixed main objective')
    values=components(state,graph)
    values.update(module_components(state, graph))
    values['module_overlap'] = values['module_overlap_counted']
    base=values['geometry'].mean()
    added=LOSS_WEIGHT*values['module_overlap'].mean()
    total=base+added
    return {'total':total, **{key:value.mean() for key,value in values.items()},
             'base_objective':base, 'weighted_module_objective':added,
            'supervision':values['local'].mean()}
