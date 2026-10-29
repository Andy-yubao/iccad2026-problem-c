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
    if state.shape[-1]!=6:raise ValueError('Training/full-dev Loss needs BOTH public source predictions')
    a=source_terms(state[...,:3],graph);b=source_terms(state[...,3:],graph)
    mean=(a['total']+b['total'])/2;worst=torch.maximum(a['total'],b['total'])
    v=dict(a);v['P42_total']=a['total'];v['P74_total']=b['total']
    v['P74_true_intersection']=b['true_intersection'];v['P74_anchored_corridor']=b['anchored_corridor'];v['P74_geometry']=b['geometry']
    v['source_loss_gap']=(a['total']-b['total']).abs()
    v['total']=mean+.25*(worst-mean)
    return v
def losses(state,graph,targets,arm,epoch=0,*,include_quality=True,start_epoch=None):
    return {k:v.mean() for k,v in per_case(state,graph,arm).items()}
