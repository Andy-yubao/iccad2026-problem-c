"""E10 geometry objective; occupancy is needed only for fixed model selection."""
from .losses import components, all_components, objective_loss, WEIGHTS


def weights(arm, epoch, start_epoch=None):
    ramp=min(1.,max(0.,epoch)/2.)
    return {key:value*(ramp if key in ('overlap','contact','boundary','mib','bbox') else 1.)
            for key,value in WEIGHTS.items()}


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    # Removing unused occupancy from training changes no geometry-loss gradient.
    values=all_components(state,graph) if include_quality else components(state,graph)
    total=objective_loss(values,'geometry',epoch)
    return {'total':total, **{key:value.mean() for key,value in values.items()},
            'supervision':values['local'].mean(), 'quality_proxy':values['geometry'].mean()}
