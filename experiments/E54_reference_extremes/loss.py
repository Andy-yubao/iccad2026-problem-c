"""Complementary public-only / reference-only fixed objectives.

No-reference execution takes the public input whitelist before evaluating any
term, so deleting or corrupting reference/target cannot influence its objective.
"""
import torch
import torch.nn.functional as F
from .batches import decode, inputs_only
from .losses import average, side_bits
from .module_loss import components as module_components
from .loss_config import configuration
from .ops import product2

def weights(arm, epoch, start_epoch=None):
    return configuration(arm)['weights']

def components(state, graph, arm):
    configuration(arm)
    g=inputs_only(graph) if arm=='no_reference' else graph
    valid=g['valid'];mask=g['mask'];area=g['area'];local=area.clamp_min(1e-12).sqrt()
    p=decode(state,g);pc=p[...,:2]+p[...,2:]/2
    pd=pc[:,:,None]-pc[:,None,:];scale=(local[:,:,None]+local[:,None,:])/2
    n=state.shape[1]
    pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
    pairs=pairs&(mask[...,0].bool()[:,:,None]|mask[...,0].bool()[:,None,:])
    margin=pd.abs()-(p[:,:,None,2:]+p[:,None,:,2:])/2
    lo=p[...,:2].masked_fill(~valid[...,None],1e4).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
    bbox=product2(hi-lo).clamp_min(1e-8)
    if arm=='no_reference':
        nearest=torch.where(margin[...,0]>=margin[...,1],margin[...,0],margin[...,1])
        penetration=(-nearest).clamp_min(0)/scale.clamp_min(1e-8)
        out={'overlap':(penetration.square()*pairs).sum((1,2))/valid.sum(1).clamp_min(1)}
        side=torch.stack((p[...,0]-lo[:,None,0],hi[:,None,0]-p[...,0]-p[...,2],
                          hi[:,None,1]-p[...,1]-p[...,3],p[...,1]-lo[:,None,1]),-1)
        required=((g['constraints'][...,4,None]&side_bits(state.device))!=0)&valid[...,None]
        out['boundary']=average(F.smooth_l1_loss(side/local[...,None],torch.zeros_like(side),reduction='none',beta=.1),required,(1,2))
        compatible=(area[:,:,None].clamp_min(1e-12).log()-area[:,None,:].clamp_min(1e-12).log()).abs()<.019
        mib=(g['mib']>0)&compatible&pairs;logs=p[...,2:].clamp_min(1e-8).log()
        out['mib']=average((logs[:,:,None]-logs[:,None,:]).square().mean(-1),mib,(1,2))
        out.update(module_components(state,g))
        out['compactness']=(bbox/(area*valid).sum(1).clamp_min(1e-8)).clamp_min(1).log().square()
        return out
    target=g['target'];r=g['reference'];rc=r[...,:2]+r[...,2:]/2
    rd=rc[:,:,None]-rc[:,None,:];rm=rd.abs()-(r[:,:,None,2:]+r[:,None,:,2:])/2
    center_error=((state[...,:2]-target[...,:2])/local[...,None]).abs().clamp_min(0)
    out={'center':average(F.smooth_l1_loss(center_error,torch.zeros_like(center_error),reduction='none',beta=.1),mask[...,:2],(1,2)),
         'shape':average((state[...,2]-target[...,2]).abs().clamp_min(0).square(),mask[...,2],(1,))}
    near=(rm.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)&pairs
    rel_error=((pd-rd)/scale.clamp_min(1e-8)[...,None]).abs().clamp_min(0)
    out['relative']=average(F.smooth_l1_loss(rel_error,torch.zeros_like(rel_error),reduction='none',beta=.1).mean(-1),near,(1,2))
    cx=(rm[...,0].abs()<1e-6)&(rm[...,1]<-1e-6)&(g['group']>0)&pairs
    cy=(rm[...,1].abs()<1e-6)&(rm[...,0]<-1e-6)&(g['group']>0)&pairs
    tx=(margin[...,0]/scale.clamp_min(1e-8)).square()+(margin[...,1]/scale.clamp_min(1e-8)+.01).clamp_min(0).square()
    ty=(margin[...,1]/scale.clamp_min(1e-8)).square()+(margin[...,0]/scale.clamp_min(1e-8)+.01).clamp_min(0).square()
    out['contact']=(tx*cx+ty*cy).sum((1,2))/(cx.sum((1,2))+cy.sum((1,2))).clamp_min(1)
    reflo=r[...,:2].masked_fill(~valid[...,None],1e4).amin(1)
    refhi=(r[...,:2]+r[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
    out['bbox']=(bbox/(refhi-reflo).prod(-1).clamp_min(1e-8)).log().square()
    return out

def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    values=components(state,graph,arm);w=configuration(arm)['weights']
    total=sum(weight*values[name].mean() for name,weight in w.items())
    return {'total':total,**{name:value.mean() for name,value in values.items()}}
