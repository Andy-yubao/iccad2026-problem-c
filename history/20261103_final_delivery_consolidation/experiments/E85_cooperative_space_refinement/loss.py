"""Identical two-source robust mean Loss for backward/monitoring/full-dev/best."""
import torch
from .losses import components
from .loss_config import configuration
from .topology import details
from .box import expansion
from .intersection import fraction
from .anchored_paths import corridor

def weights(arm,epoch,start_epoch=None):return configuration(arm)['weights']
def source_terms(predicted,graph):
    v=components(predicted,graph);v['common_original_G']=v['geometry'];v['center_objective']=v['center_local']
    v['collision_topology'],diagnostic=details(predicted,graph);v['topology_unconditional']=diagnostic['unconditional']
    v['box_expansion']=expansion(predicted,graph);v['true_intersection']=fraction(predicted,graph)
    v['anchored_corridor'],_=corridor(predicted,graph)
    v['total']=v['geometry']+4*v['true_intersection']+5*v['box_expansion']+.25*v['anchored_corridor']
    v['supervision']=v['local']+5*v['box_expansion']
    return v
def per_case(state,graph,arm):
    configuration(arm)
    if state.shape[-1]!=3:raise ValueError('Single frozen E74 prediction required')
    return source_terms(state,graph)

def losses(state,graph,targets,arm,epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
