"""Only the training overlap coefficient changes; selection keeps E17 weights."""
from .losses import components, all_components, objective_loss, WEIGHTS

TRAIN_OVERLAP_WEIGHT = 1.0
SELECTION_OVERLAP_WEIGHT = WEIGHTS['overlap']


def weights(arm, epoch, start_epoch=None):
    ramp=min(1.,max(0.,epoch)/2.)
    result={key:value*(ramp if key in ('overlap','contact','boundary','mib','bbox') else 1.)
            for key,value in WEIGHTS.items()}
    result['overlap']=TRAIN_OVERLAP_WEIGHT*ramp
    return result


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    # WEIGHTS and repair stay frozen so best selection is comparable with E17.
    values=all_components(state,graph) if include_quality else components(state,graph)
    ramp=min(1.,max(0.,epoch)/2.)
    total=objective_loss(values,'geometry',epoch)+(TRAIN_OVERLAP_WEIGHT-SELECTION_OVERLAP_WEIGHT)*ramp*values['overlap'].mean()
    return {'total':total, **{key:value.mean() for key,value in values.items()},
            'supervision':values['local'].mean(), 'quality_proxy':values['geometry'].mean()}
