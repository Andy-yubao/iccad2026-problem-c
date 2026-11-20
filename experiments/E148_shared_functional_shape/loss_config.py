"""One joint objective: readouts4/8/12, weights.1/.1/.8, no overlap only4."""
from .loss import DEFAULT,configuration as objective_configuration
ARMS=('e148_shared_functional_shape',)
SUPERVISED_LAYERS=(4,8,12)
NO_OVERLAP_LAYERS=(4,)

def layer_weights(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {4:.1,8:.1,12:.8}

def configuration(arm):
    value=objective_configuration(DEFAULT)
    value.update(version='E148.joint.4_8_12.v1',arm=arm,layer_weights={str(k):v for k,v in layer_weights(arm).items()},
        no_overlap_layers=[4],full_loss_layers=[8,12],
        removed_group='Only readout4: A*(penetration + module_load_mean + module_max_excess10)',
        selection='Same fixed aggregate for backward/monitor/best/LR/earlystop; no Q selection',training_authorized=True)
    return value
