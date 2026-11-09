"""Same three-group objective for backward, monitors, best, LR and evaluation."""
from .loss_config import configuration
from .simple_loss import components

def weights(arm,epoch=0,start_epoch=None):return configuration(arm)['weights']

def per_case(state,graph,arm):
    w=weights(arm)
    return components(state,graph,{k:w[k] for k in ('overlap','task')})

def losses(state,graph,targets=None,arm='simple',epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
