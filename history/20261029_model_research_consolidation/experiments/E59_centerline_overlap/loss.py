from .losses import components,WEIGHTS
from .loss_config import configuration
def weights(arm,epoch,start_epoch=None):return configuration(arm)['weights']
def losses(state,graph,targets,arm,epoch,*,include_quality=True,start_epoch=None):
    c=configuration(arm)
    values=components(state,graph,overlap_kind=c['overlap_kind'],overlap_kappa=c['overlap_kappa'])
    total=sum(WEIGHTS[k]*values[k].mean() for k in WEIGHTS)
    return {'total':total,**{k:v.mean() for k,v in values.items()},'supervision':values['local'].mean()}
