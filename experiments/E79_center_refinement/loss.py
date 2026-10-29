"""Same fixed center objective for training/monitoring/selection/full dev."""
from .losses import components
from .loss_config import configuration
from .topology import details
from .box import expansion
from .intersection import fraction
from .source_guard import regression

def weights(arm,epoch,start_epoch=None):return configuration(arm)['weights']

def per_case(state,graph,arm):
    if state.shape[-1]!=6:raise ValueError('E79 needs predicted3 plus detached public source3')
    predicted=state[...,:3];source=state[...,3:].detach();config=configuration(arm)
    v=components(predicted,graph);v['common_original_G']=v['geometry'];v['center_objective']=v['center_local']
    v['collision_topology'],diagnostic=details(predicted,graph);v['topology_unconditional']=diagnostic['unconditional']
    v['box_expansion']=expansion(predicted,graph);v['true_intersection']=fraction(predicted,graph)
    v['source_true_intersection']=fraction(source,graph)
    v['topology_regression'],v['topology_flip_fraction']=regression(predicted,source,graph)
    w=config['weights'];v['total']=v['geometry']+w['true_intersection']*v['true_intersection']+w['box_expansion']*v['box_expansion']+w['topology_regression']*v['topology_regression']
    v['supervision']=v['local']+5*v['box_expansion']
    return v

def losses(state,graph,targets,arm,epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
