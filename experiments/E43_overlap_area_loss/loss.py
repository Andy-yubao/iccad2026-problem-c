"""Geometry plus pair-area training; original geometry+occupancy selection."""
from .losses import components, all_components, objective_loss, WEIGHTS
from .area_loss import AREA_WEIGHT, overlap_area


def weights(arm, epoch, start_epoch=None):
    ramp=min(1.,max(0.,epoch)/2.)
    return {**{key:value*(ramp if key in ('overlap','contact','boundary','mib','bbox') else 1.)
            for key,value in WEIGHTS.items()}, 'overlap_area': AREA_WEIGHT*ramp}


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    values=all_components(state,graph) if include_quality else components(state,graph)
    if not include_quality:
        values['overlap_area'] = overlap_area(state, graph)
    base=objective_loss(values,'geometry',epoch)
    added=weights(arm,epoch,start_epoch)['overlap_area']*values['overlap_area'].mean()
    total=base+added
    return {'total':total, **{key:value.mean() for key,value in values.items()},
            'base_objective':base, 'weighted_area_objective':added,
            'supervision':values['local'].mean(), 'quality_proxy':values['geometry'].mean()}
