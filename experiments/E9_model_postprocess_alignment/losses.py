"""Geometry supervision computed from geometry, with no relation outputs."""
import torch
import torch.nn.functional as F
from .batches import decode

WEIGHTS={'center_local':1.,'shape':.25,'relative_local':.25,'overlap':.5,
         'contact':.5,'boundary':.25,'mib':.1,'bbox':.1}


def average(x,mask,dims):
    return (x*mask).sum(dims)/mask.sum(dims).clamp_min(1)


def components(state,g):
    mask=g['mask'];valid=g['valid'];target=g['target'];b,n,_=state.shape
    error=(state-target).square()
    out={'mse':average(error,mask,(1,2)),
         'center_mse':average(error[...,:2],mask[...,:2],(1,2)),
         'shape_mse':average(error[...,2],mask[...,2],(1,))}
    local=g['area'].clamp_min(1e-12).sqrt()
    out['center_local']=average(F.smooth_l1_loss(state[...,:2]/local[...,None],target[...,:2]/local[...,None],reduction='none',beta=.1),mask[...,:2],(1,2))
    out['shape']=out['shape_mse']
    p=decode(state,g);r=g['reference'];pc=p[...,:2]+p[...,2:]/2;rc=r[...,:2]+r[...,2:]/2
    pd=pc[:,:,None,:]-pc[:,None,:,:];rd=rc[:,:,None,:]-rc[:,None,:,:]
    ps=(local[:,:,None]+local[:,None,:])/2
    pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
    mutable=(mask[...,0].bool()[:,:,None]|mask[...,0].bool()[:,None,:])
    pairs=pairs&mutable
    extent=(p[:,:,None,2:]+p[:,None,:,2:])/2
    rextent=(r[:,:,None,2:]+r[:,None,:,2:])/2
    margin=pd.abs()-extent;rm=rd.abs()-rextent
    near=(rm.clamp_min(0).square().sum(-1).sqrt()<=.25*ps)&pairs
    rel=F.smooth_l1_loss(pd/ps.clamp_min(1e-8)[...,None],rd/ps.clamp_min(1e-8)[...,None],reduction='none',beta=.1).mean(-1)
    out['relative_local']=average(rel,near,(1,2))
    penetration=(-margin.max(-1).values).clamp_min(0)/ps.clamp_min(1e-8)
    # Sum per layout, normalize by blocks, not all pairs (which dilutes collisions).
    out['overlap']=(penetration.square()*pairs).sum((1,2))/valid.sum(1).clamp_min(1)
    # Reference contacts are train supervision only; inferred at evaluation, never input.
    cx=(rm[...,0].abs()<1e-6)&(rm[...,1]<-1e-6)&(g['group']>0)&pairs
    cy=(rm[...,1].abs()<1e-6)&(rm[...,0]<-1e-6)&(g['group']>0)&pairs
    tx=(margin[...,0]/ps.clamp_min(1e-8)).square()+(margin[...,1]/ps.clamp_min(1e-8)+.01).clamp_min(0).square()
    ty=(margin[...,1]/ps.clamp_min(1e-8)).square()+(margin[...,0]/ps.clamp_min(1e-8)+.01).clamp_min(0).square()
    out['contact']=(tx*cx+ty*cy).sum((1,2))/(cx.sum((1,2))+cy.sum((1,2))).clamp_min(1)
    lo=p[...,:2].masked_fill(~valid[...,None],1e4).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
    side=torch.stack((p[...,0]-lo[:,None,0],hi[:,None,0]-p[...,0]-p[...,2],
                      hi[:,None,1]-p[...,1]-p[...,3],p[...,1]-lo[:,None,1]),-1)
    required=(g['constraints'][...,4,None]&torch.tensor([1,2,4,8],device=state.device))!=0
    required=required&valid[...,None]
    out['boundary']=average(F.smooth_l1_loss(side/local[...,None],torch.zeros_like(side),reduction='none',beta=.1),required,(1,2))
    compatible=(g['area'][:,:,None].clamp_min(1e-12).log()-g['area'][:,None,:].clamp_min(1e-12).log()).abs()<.019
    mib=(g['mib']>0)&compatible&pairs
    logs=p[...,2:].clamp_min(1e-8).log()
    diff=(logs[:,:,None,:]-logs[:,None,:,:]).square().mean(-1)
    out['mib']=average(diff,mib,(1,2))
    reflo=r[...,:2].masked_fill(~valid[...,None],1e4).amin(1)
    refhi=(r[...,:2]+r[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
    bbox=(hi-lo).prod(-1).clamp_min(1e-8);refbbox=(refhi-reflo).prod(-1).clamp_min(1e-8)
    out['bbox']=(bbox/refbbox).log().square()
    out['local']=out['center_local']+.25*out['shape']+.25*out['relative_local']
    out['geometry']=sum(WEIGHTS[k]*out[k] for k in WEIGHTS)
    return out


ALIGNED_WEIGHTS={'wire_excess':.25,'area_excess':.1,'boundary_extra':.5}


def weighted_wire(positions,g):
    """Exact weighted Manhattan centers in normalized units; training proxy only."""
    centers=positions[...,:2]+positions[...,2:]/2
    batch=torch.arange(positions.shape[0],device=positions.device)[:,None]
    edges=g['wire_b2b'];valid=(edges[...,:2]>=0).all(-1)
    i=edges[...,0].long().clamp_min(0);j=edges[...,1].long().clamp_min(0)
    wire=((centers[batch,i]-centers[batch,j]).abs().sum(-1)*edges[...,2].clamp_min(0)*valid).sum(1)
    edges=g['wire_p2b'];valid=(edges[...,:2]>=0).all(-1)
    i=edges[...,0].long().clamp_min(0);j=edges[...,1].long().clamp_min(0)
    wire+=((g['pins'][batch,i]-centers[batch,j]).abs().sum(-1)*edges[...,2].clamp_min(0)*valid).sum(1)
    return wire


def all_components(state,g):
    out=components(state,g)
    positions=decode(state,g);valid=g['valid']
    lo=positions[...,:2].masked_fill(~valid[...,None],float('inf')).amin(1)
    hi=(positions[...,:2]+positions[...,2:]).masked_fill(~valid[...,None],-float('inf')).amax(1)
    reflo=g['reference'][...,:2].masked_fill(~valid[...,None],float('inf')).amin(1)
    refhi=(g['reference'][...,:2]+g['reference'][...,2:]).masked_fill(~valid[...,None],-float('inf')).amax(1)
    out['area_excess']=((hi-lo).prod(-1)/(refhi-reflo).prod(-1).clamp_min(1e-8)-1).clamp_min(0)
    wire=weighted_wire(positions,g);refwire=weighted_wire(g['reference'],g)
    weight=sum(g[key][...,2].clamp_min(0).sum(1) for key in ('wire_b2b','wire_p2b')).clamp_min(1e-8)
    out['wire_excess']=torch.where(refwire>1e-6,(wire/refwire.clamp_min(1e-6)-1).clamp_min(0),wire/weight)
    # Retain geometry supervision and collision penalties. Stop penalizing a
    # bbox smaller than the reference; strengthen public boundary coordination.
    out['aligned']=out['geometry']-WEIGHTS['bbox']*out['bbox']
    out['aligned']+=ALIGNED_WEIGHTS['area_excess']*out['area_excess']
    out['aligned']+=ALIGNED_WEIGHTS['wire_excess']*out['wire_excess']
    out['aligned']+=ALIGNED_WEIGHTS['boundary_extra']*out['boundary']
    return out


def objective_loss(values,variant,epoch):
    if variant in ('geometry_control','center_refiner'):return values['geometry'].mean()
    if variant!='quality_aligned':raise ValueError(variant)
    ramp=min(1.,max(0.,epoch)/2.)
    return (values['geometry']+ramp*(values['aligned']-values['geometry'])).mean()

