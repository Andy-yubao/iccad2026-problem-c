from pathlib import Path
p=Path('experiments/E107_geometry_posttrain')
files={
'geometry.py':'''"""Exact rectangle-union/coverage via sorted event cells, differentiable a.e."""
import torch
from .batches import decode

def measures(state,g):
    p=decode(state,g);valid=g['valid'];lo=p[...,:2];hi=lo+p[...,2:]
    x=torch.cat((lo[...,0],hi[...,0]),-1).sort(-1,stable=True).values
    y,yorder=torch.cat((lo[...,1],hi[...,1]),-1).sort(-1,stable=True)
    dx=x[:,1:]-x[:,:-1];dy=y[:,1:]-y[:,:-1]
    xm=(x[:,1:]+x[:,:-1])/2;ym=(y[:,1:]+y[:,:-1])/2
    ax=(xm[:,:,None]>=lo[:,None,:,0])&(xm[:,:,None]<hi[:,None,:,0])&valid[:,None,:]
    ay=(ym[:,:,None]>=lo[:,None,:,1])&(ym[:,:,None]<hi[:,None,:,1])&valid[:,None,:]
    events=torch.cat((ax.to(torch.int32),-ax.to(torch.int32)),-1)
    events=events.gather(-1,yorder[:,None,:].expand(-1,events.shape[1],-1))
    depth=events.cumsum(-1,dtype=torch.int32)[...,:-1]
    cell=dx[:,:,None]*dy[:,None,:]
    union=(cell*(depth>0)).sum((1,2));overlap_cell=cell*(depth>1)
    overlap_region=overlap_cell.sum((1,2))
    covered=(torch.bmm(overlap_cell,ay.to(p.dtype))*ax.to(p.dtype)).sum(1)
    area=g['area'].clamp_min(1e-12);total=(area*valid).sum(-1).clamp_min(1e-12)
    fraction=(covered/area).clamp(0,1)*valid;count=valid.sum(-1).clamp_min(1)
    bblo=lo.masked_fill(~valid[...,None],float('inf')).amin(1)
    bbhi=hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
    bbox=(bbhi-bblo).prod(-1).clamp_min(1e-12);excess=(fraction-.1).clamp_min(0)
    return dict(area_excess=(1-union/total).clamp_min(0),overlap_region_fraction=overlap_region/total,
        tail=((excess/(excess+.05)+excess)*valid).sum(-1)/count,
        whitespace=(1-union/bbox).clamp_min(0),covered_fraction=fraction,
        N10=((fraction>.1)&valid).sum(-1).to(p.dtype),
        N10_fraction=((fraction>.1)&valid).sum(-1)/count,max_covered_fraction=fraction.amax(-1),
        union_area=union,bbox_over_A=bbox/total,positions=p)
''',
'runtime.py':'''"""Frozen E105 parent, no optimized tensors or reference inputs."""
import json
import torch
from .io import HERE,sha
from .model import GeometryModel

def parent_model():
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'));path=HERE/plan['parent_checkpoint']
    if sha(path)!=plan['parent_checkpoint_sha256']:raise ValueError('Frozen parent drift')
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'],strict=True)
    model.requires_grad_(False)
    return model
''',
'guards.py':'''"""Same-case frozen-parent relative gates plus original absolute numerical gates."""
import torch

def safe(total,parent_total,norm=None,finite=None):
    ok=torch.isfinite(total).all()&torch.isfinite(parent_total).all()
    if finite is not None:ok=ok&finite
    if norm is not None:ok=ok&torch.isfinite(norm)&(norm<10000.)
    mean=total.mean();base_mean=parent_total.mean()
    case_limit=torch.maximum(10*parent_total,parent_total+20.)
    mean_limit=torch.maximum(3*base_mean,base_mean+5.)
    return ok&(total.amax()<1000.)&(mean<1000.)&(total<case_limit).all()&(mean<mean_limit)

def details(total,parent_total,norm=None):
    return dict(mean_total=float(total.mean()),max_case=float(total.max()),
        parent_mean=float(parent_total.mean()),parent_max=float(parent_total.max()),
        relative_mean_limit=float(torch.maximum(3*parent_total.mean(),parent_total.mean()+5.)),
        max_relative_case_ratio=float((total/parent_total.clamp_min(1e-6)).max()),
        gradient_norm=None if norm is None else float(norm),absolute_case_limit=1000.,gradient_limit=10000.)
''',
'loss_config.py':'''"""Fixed new objective, normalization frozen from train500 only."""
import json
from functools import lru_cache
from .io import HERE
A=.08975162861857286
B=.3336832108335229
COEFFICIENTS={'position':.1,'shape':.025,'area_excess_scaled':1.,'tail_scaled':1.,'whitespace_scaled':1.,
    'soft_original':2*B,'soft_barrier':2.,'wirelength':.5*B,'wire_barrier':1.,'trust':1.}
SOFT_TOLERANCES=(.01,.01,.0025)

@lru_cache(maxsize=1)
def normalization():
    data=json.loads((HERE/'normalization.json').read_text(encoding='utf-8'))
    if data['status']!='frozen_train500':raise ValueError('Normalization not frozen')
    return data

def configuration():
    return dict(revision='E107.union_geometry.ref10.v1',coefficients=dict(COEFFICIENTS),
        normalization=normalization(),soft_tolerances=list(SOFT_TOLERANCES),
        trust_center_radius=.1,trust_log_aspect_radius=.1,tail_threshold=.1,tail_smooth_width=.05,
        aggregation='layout terms then arithmetic mean',reference_position=.1,reference_shape=.025,
        monitoring='same fixed loss; eval/no augmentation; trust always deterministic student eval')
''',
'loss.py':'''"""Geometry-first loss, exact unions, frozen-parent preservation and local trust."""
import torch
from .simple_loss import components
from .geometry import measures
from .batches import decode,inputs_only
from .loss_config import A,B,COEFFICIENTS,SOFT_TOLERANCES,normalization

def raw_terms(state,g,parent_state,clean_state=None):
    old=components(state,g,{'overlap':A,'task':B});parent=components(parent_state,g,{'overlap':A,'task':B})
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
    out.update(position=old['position'],shape=old['shape'],wirelength=old['wirelength'],
        soft_original=old['boundary_contribution']+old['grouping_contribution']+old['mib_contribution'],
        boundary=old['boundary'],grouping=old['grouping'],mib=old['mib'],soft_barrier=barrier,
        wire_barrier=wire_barrier,trust=trust,legacy_E105_total=old['total'])
    return out

def per_case(state,g,parent_state,clean_state=None):
    out=raw_terms(state,g,parent_state,clean_state);scales=normalization()['scales']
    for key in ('area_excess','tail','whitespace'):out[key+'_scaled']=out[key]/scales[key]
    out['total']=sum(value*out[key] for key,value in COEFFICIENTS.items())
    return out

def compute(model,parent,g,training=False):
    public=inputs_only(g)
    with torch.no_grad():anchor=parent(public)
    model.eval();clean=model(public)
    if training:model.train();state=model(public)
    else:state=clean
    values=per_case(state,g,anchor,clean)
    with torch.no_grad():baseline=per_case(anchor,g,anchor,anchor)
    return state,values,baseline
'''
}
for name,source in files.items():(p/name).write_text(source,encoding='utf-8')
print('Written geometry, loss, frozen parent and strict guards')
