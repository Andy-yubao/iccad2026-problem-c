"""One fixed total for backward, monitor, best, LR and early stopping."""
from .losses import components
from .loss_config import configuration

def weights(arm,epoch=0,start_epoch=None):return configuration(arm)['weights']
def per_case(state,graph,arm):
    configuration(arm);return components(state,graph)
def losses(state,graph,targets,arm,epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
