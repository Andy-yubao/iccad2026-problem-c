"""Four frozen cells; terminal.8 plus auxiliary total.2 in all cells."""
from .loss import A,B,DEFAULT,configuration as objective_configuration
ARMS=('e119_stride3','e119_stride2','e120_stride3','e120_stride2')

def layer_weights(arm):
    if arm not in ARMS:raise ValueError(arm)
    auxiliary=(3,6,9) if arm.endswith('3') else (2,4,6,8,10)
    return {**{layer:.2/len(auxiliary) for layer in auxiliary},12:.8}

def configuration(arm):
    weights=layer_weights(arm)
    value=objective_configuration(DEFAULT)
    value.update(arm=arm,layer_weights={str(k):v for k,v in weights.items()},training_authorized=False,
                 selection='Unique own fixed aggregate Loss best; never Q',
                 common_final_new='final_total diagnostic only',common_final_original='original_final_total diagnostic only')
    return value
