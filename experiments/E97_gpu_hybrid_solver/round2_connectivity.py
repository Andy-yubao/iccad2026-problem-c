"""Current-geometry group MST task; discrete tree detached, edges differentiated."""
import torch
import torch.nn.functional as F
from .candidate3 import per_case as c3_per_case,Packer,attach as base_attach
from .train_io.data import LEGACY
from .nn85.base_model import decode

_BITS={}
MAX_GROUP_SIZE=10  # public inventory of registered train16k/dev500; not a deployment limit
def network_inputs(g):return {k:g[k] for k in LEGACY}

def attach(rows,labels,indices=None):
    base_attach(rows,labels,indices)
    for row in rows:
        groups=row['graph']['constraints'][:,3]
        assert all(int((groups==v).sum())<=MAX_GROUP_SIZE for v in groups.unique() if v>0)
    return rows

@torch.no_grad()
def mst_mask(cost,group,valid):
    n=cost.shape[-1];ix=torch.arange(n,device=cost.device)
    same=(group[:,:,None]==group[:,None,:])&(group[:,:,None]>0)&valid[:,:,None]&valid[:,None,:]
    root=torch.where(same,ix[None,None,:],n).amin(-1)
    visited=(ix[None,:]==root)&valid&(group>0);chosen=torch.zeros_like(same)
    for _ in range(MAX_GROUP_SIZE-1):
        allowed=same&visited[:,:,None]&~visited[:,None,:]
        distance,source=cost.masked_fill(~allowed,1e6).min(1)
        target=distance[:,None,:].expand_as(cost).masked_fill(~same,1e6).argmin(-1)
        add=(ix[None,:]==target)&~visited&valid&(group>0)&(distance<1e6)
        edge=(ix[None,:,None]==source[:,None,:])&add[:,None,:]
        chosen=chosen|edge;visited=visited|add
    return chosen

def public_denominator(c,valid):
    n=valid.shape[-1];ix=torch.arange(n,device=valid.device);count=(valid&(c[...,4]>0)).sum(-1)
    for col in (2,3):
        value=c[...,col];same=(value[:,:,None]==value[:,None,:])&(value[:,:,None]>0)&valid[:,:,None]&valid[:,None,:]
        root=torch.where(same,ix[None,None,:],n).amin(-1)
        count=count+((value>0)&valid&(ix[None,:]!=root)).sum(-1)
    return count.clamp_min(1)

def raw_terms(state,g,target,usable):
    out=c3_per_case(state,g,target,usable);p=decode(state,g);valid=g['valid'];local=g['area'].clamp_min(1e-12).sqrt();c=g['constraints']
    lo=p[...,:2];hi=lo+p[...,2:];minimum=lo.masked_fill(~valid[...,None],1e4).amin(1);maximum=hi.masked_fill(~valid[...,None],-1e4).amax(1)
    side=torch.stack((lo[...,0]-minimum[:,None,0],maximum[:,None,0]-hi[...,0],maximum[:,None,1]-hi[...,1],lo[...,1]-minimum[:,None,1]),-1)/local[...,None]
    if state.device not in _BITS:_BITS[state.device]=torch.tensor([1,2,4,8],device=state.device)
    bits=_BITS[state.device];required=((c[...,4,None]&bits)!=0)&valid[...,None]
    boundary=torch.where(required,F.smooth_l1_loss(side,torch.zeros_like(side),beta=.1,reduction='none'),0.).amax(-1).sum(-1)
    center=lo+p[...,2:]/2;delta=(center[:,:,None]-center[:,None,:]).abs();extent=(p[...,2:][:,:,None]+p[...,2:][:,None,:])/2
    normal=(delta/extent.clamp_min(1e-8)).clamp(0,8)
    touching=F.smooth_l1_loss(normal,torch.ones_like(normal),beta=.1,reduction='none')
    pair_local=(local[:,:,None]+local[:,None,:])/2
    tangent=torch.relu(.01-(extent-delta)/pair_local.clamp_min(1e-8)[...,None]).square()
    cost=torch.minimum(touching[...,0]+tangent[...,1],touching[...,1]+tangent[...,0])
    tree=mst_mask(cost.detach(),c[...,3],valid)
    denom=public_denominator(c,valid)
    out['public_boundary']=boundary/denom;out['public_group_contact']=(cost*tree).sum((1,2))/denom
    out['public_soft']=out['public_boundary']+out['public_group_contact']
    return out
