"""Single fixed new objective for backward, monitoring, selection and schedule."""
from .loss import A,B,DEFAULT,configuration as objective_configuration
ARMS=('global2',)
def layer_weights(arm):
    if arm!='global2':raise ValueError(arm)
    return {4:.2,12:.8}
def configuration(arm):
    layer_weights(arm)
    out=objective_configuration(DEFAULT)
    out.update(arm=arm,training_authorized=True,selection='unique own-new-Loss dev500 best; never Q',common_original_final='original_final_total diagnostic only')
    return out
