"""Fixed G plus direction-aware near-pair supervision."""
from .losses import components, WEIGHTS
from .loss_config import configuration
from .topology import details

def weights(arm,epoch,start_epoch=None): return configuration(arm)['weights']
def per_case(state,graph,arm):
    config=configuration(arm);v=components(state,graph)
    v['common_original_G']=v['geometry'];v['center_objective']=v['center_local']
    v['collision_topology'],diagnostic=details(state,graph)
    v['topology_unconditional']=diagnostic['unconditional']
    v['total']=v['geometry']+config['weights']['collision_topology']*v['collision_topology']
    v['supervision']=v['local']+.5*v['collision_topology']
    return v
def losses(state,graph,targets,arm,epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
