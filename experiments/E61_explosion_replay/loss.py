"""Original fixed eight-term G, copied semantics from E57."""
from .losses import components, WEIGHTS

def weights(arm,epoch,start_epoch=None):return dict(WEIGHTS)

def losses(state,graph,targets,arm,epoch,**kwargs):
    values=components(state,graph)
    total=sum(WEIGHTS[k]*values[k].mean() for k in WEIGHTS)
    return {'total':total,**{k:v.mean() for k,v in values.items()},'supervision':values['local'].mean()}
