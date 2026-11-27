"""Same current-epoch objective for backward and train/dev monitoring."""
from .losses import components
from .loss_config import configuration
from .curriculum import position_weight

def weights(arm,epoch=200,start_epoch=None):
    result=configuration(arm)['weights'].copy()
    result['position']=position_weight(epoch)
    return result
def per_case(state,graph,arm,epoch=200,*,position_coefficient=None):
    configuration(arm)
    coefficient=position_weight(epoch) if position_coefficient is None else position_coefficient
    return components(state,graph,position_weight=coefficient)
def losses(state,graph,targets,arm,epoch=200,*,include_quality=True,start_epoch=None,position_coefficient=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm,epoch,
                        position_coefficient=position_coefficient).items()}
