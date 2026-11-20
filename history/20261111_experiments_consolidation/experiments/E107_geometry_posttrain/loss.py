"""Geometry-first loss, exact unions, frozen-parent preservation and local trust."""
import torch
from .simple_loss import components
from .geometry import measures
from .batches import decode,inputs_only
from .loss_config import A,B,COEFFICIENTS,SOFT_TOLERANCES,normalization

def raw_terms(state,g,parent_state,clean_state=None,parent_terms=None,student_terms=None):
    old=components(state,g,{'overlap':A,'task':B}) if student_terms is None else student_terms
    parent=components(parent_state,g,{'overlap':A,'task':B}) if parent_terms is None else parent_terms
    geom=measures(state,g);valid=g['valid']
    clean=decode(state if clean_state is None else clean_state,g);anchor=decode(parent_state,g)
    c=clean[...,:2]+clean[...,2:]/2;ac=anchor[...,:2]+anchor[...,2:]/2
    delta=((c-ac).square().sum(-1)+1e-12).sqrt()/g['area'].clamp_min(1e-12).sqrt()
    ratio=clean[...,2:].clamp_min(1e-12).log();aratio=anchor[...,2:].clamp_min(1e-12).log()
    dr=((ratio[...,0]-ratio[...,1])-(aratio[...,0]-aratio[...,1])).abs()
    movable=g['mask'][...,0]*valid;shape_free=g['mask'][...,2]*valid
    trust=(((delta/.1-1).clamp_min(0).square()*movable).sum(-1)/movable.sum(-1).clamp_min(1)
        +((dr/.1-1).clamp_min(0).square()*shape_free).sum(-1)/shape_free.sum(-1).clamp_min(1))
    barrier=torch.zeros_like(trust)
    for key,tolerance in zip(('boundary','grouping','mib'),SOFT_TOLERANCES):
        barrier=barrier+(old[key]/parent[key].clamp_min(tolerance)-1).clamp_min(0).square()
    wire_barrier=(old['wirelength']/parent['wirelength'].clamp_min(1e-4)-1.05).clamp_min(0).square()
    out={key:geom[key] for key in ('area_excess','overlap_region_fraction','tail','whitespace','N10','N10_fraction','max_covered_fraction','bbox_over_A')}
    out.update(penetration=old['penetration'],position=old['position'],shape=old['shape'],wirelength=old['wirelength'],
        soft_original=old['boundary_contribution']+old['grouping_contribution']+old['mib_contribution'],
        boundary=old['boundary'],grouping=old['grouping'],mib=old['mib'],soft_barrier=barrier,
        wire_barrier=wire_barrier,trust=trust,legacy_E105_total=old['total'])
    return out

def per_case(state,g,parent_state,clean_state=None,parent_terms=None,student_terms=None):
    out=raw_terms(state,g,parent_state,clean_state,parent_terms,student_terms);scales=normalization()['scales']
    for key in ('area_excess','tail','whitespace'):out[key+'_scaled']=out[key]/scales[key]
    out['total']=sum(value*out[key] for key,value in COEFFICIENTS.items())
    return out

def compute(model,parent,g,training=False):
    public=inputs_only(g)
    with torch.no_grad():
        anchor=parent(public)
        anchor_terms=components(anchor,g,{'overlap':A,'task':B})
    model.eval();clean=model(public)
    if training:model.train();state=model(public)
    else:state=clean
    values=per_case(state,g,anchor,clean,parent_terms=anchor_terms)
    with torch.no_grad():baseline=per_case(anchor,g,anchor,anchor,parent_terms=anchor_terms,student_terms=anchor_terms)
    return state,values,baseline
