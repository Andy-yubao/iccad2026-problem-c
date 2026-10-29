"""Fixed G plus direction-aware near-pair supervision."""
from .losses import components, WEIGHTS
from .loss_config import configuration
from .topology import details
from .box import expansion
from .intersection import fraction
from .anchored_paths import corridor

def weights(arm,epoch,start_epoch=None): return configuration(arm)['weights']
def per_case(state,graph,arm):
    config=configuration(arm);v=components(state,graph)
    v['common_original_G']=v['geometry'];v['center_objective']=v['center_local']
    v['collision_topology'],diagnostic=details(state,graph)
    v['topology_unconditional']=diagnostic['unconditional']
    v['box_expansion']=expansion(state,graph)
    v['true_intersection']=fraction(state,graph)
    v['anchored_corridor'],_=corridor(state,graph)
    v['total']=v['geometry']+config['weights']['true_intersection']*v['true_intersection']+config['weights']['box_expansion']*v['box_expansion']+config['weights']['anchored_corridor']*v['anchored_corridor']
    v['supervision']=v['local']+5*v['box_expansion']
    return v
def losses(state,graph,targets,arm,epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
