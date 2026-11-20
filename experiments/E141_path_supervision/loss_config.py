"""Frozen three-arm supervision and obstruction configuration."""
from .loss import DEFAULT,configuration as objective_configuration

ARMS=('e119_blockage','e119_layerwise','e119_combined')
FULL_LAYERS=(3,6,9,12)

def layer_weights(arm):
    if arm not in ARMS:raise ValueError(arm)
    layers=FULL_LAYERS if arm=='e119_blockage' else tuple(range(1,13))
    return {k:(.8 if k==12 else .2/(len(layers)-1)) for k in layers}

def has_blockage(arm):
    if arm not in ARMS:raise ValueError(arm)
    return arm!='e119_layerwise'

def configuration(arm):
    value=objective_configuration(DEFAULT)
    value.update(version='E141.path_supervision.v1',arm=arm,
        layer_weights={str(k):v for k,v in layer_weights(arm).items()},full_loss_layers=list(FULL_LAYERS),
        no_overlap_layers=[k for k in layer_weights(arm) if k not in FULL_LAYERS],
        blockage_enabled=has_blockage(arm),blockage_coefficient=1.,original_GB_multiplier=1.,
        same_backward_monitor_best_lr_earlystop=True,selection='unique own-Loss dev500 mean best, never Q',
        initial_S0_supervised=False,topology_similarity_in_training=False)
    return value
