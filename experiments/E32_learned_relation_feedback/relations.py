"""Small symmetric pair policy. Only public graph and actual solver states enter."""
import torch
from torch import nn
import torch.nn.functional as F
from .batches import decode

def encode_positions(positions,g):
    return torch.cat((positions[...,:2]+positions[...,2:]/2,
                      (positions[...,2]/positions[...,3].clamp_min(1e-12)).clamp_min(1e-12).log()[...,None]),-1)

class RelationPolicy(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.width=width
        self.embed=nn.Sequential(nn.Linear(29,width),nn.SiLU(),nn.LayerNorm(width))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(4*width,width),nn.SiLU(),nn.LayerNorm(width)) for _ in range(2)])
        self.pair=nn.Sequential(nn.Linear(2*width+15,width),nn.SiLU(),nn.Linear(width,3))

    def forward(self,g,draft,current):
        if set(g)-{'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Relation policy accepts only public graph fields')
        p=decode(current,g);d=decode(draft,g);valid=g['valid'];local=g['area'].sqrt().clamp_min(1e-6)
        center=p[...,:2]+p[...,2:]/2;dc=d[...,:2]+d[...,2:]/2
        delta=(center[:,:,None]-center[:,None,:]).abs();extent=(p[:,:,None,2:]+p[:,None,:,2:])/2
        gap=delta-extent;pairmask=valid[:,:,None]&valid[:,None,:]
        pairmask=pairmask&~torch.eye(current.shape[1],device=current.device,dtype=torch.bool)[None]
        penetration=(-gap.amax(-1)).clamp_min(0)*pairmask
        collision=torch.stack((penetration.sum(-1)/local,penetration.amax(-1)/local),-1)
        lo=p[...,:2].masked_fill(~valid[...,None],1e4).amin(1)
        hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
        sides=torch.stack((p[...,0]-lo[:,None,0],hi[:,None,0]-p[...,0]-p[...,2],hi[:,None,1]-p[...,1]-p[...,3],p[...,1]-lo[:,None,1]),-1)/local[...,None]
        h=self.embed(torch.cat((g['features'],draft,current,collision,sides),-1))*valid[...,None]
        for layer in self.messages:
            h=(h+layer(torch.cat((h,g['adjacency']@h,g['group']@h,g['mib']@h),-1)))*valid[...,None]
        scale=(local[:,:,None]+local[:,None,:])[...,None]/2
        geo=torch.cat((delta/scale,extent/scale,gap/scale,
            (dc[:,:,None]-dc[:,None,:]).abs()/scale,(d[:,:,None,2:]+d[:,None,:,2:])/2/scale,
            (p[:,:,None,2:].clamp_min(1e-8).log()-p[:,None,:,2:].clamp_min(1e-8).log()).abs(),
            ((g['adjacency']+g['adjacency'].transpose(1,2))/2)[...,None],
            (g['group']>0)[...,None],(g['mib']>0)[...,None]),-1)
        return self.pair(torch.cat((h[:,:,None]+h[:,None,:],(h[:,:,None]-h[:,None,:]).abs(),geo),-1))

def targets(g):
    r=g['reference'];center=r[...,:2]+r[...,2:]/2
    delta=(center[:,:,None]-center[:,None,:]).abs();extent=(r[:,:,None,2:]+r[:,None,:,2:])/2
    local=(g['area'].sqrt()[:,:,None]+g['area'].sqrt()[:,None,:])/2
    gap=(delta-extent)/local.clamp_min(1e-6)[...,None]
    valid=g['valid'][:,:,None]&g['valid'][:,None,:]
    valid=valid&~torch.eye(r.shape[1],device=r.device,dtype=torch.bool)[None]
    mutable=g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:]
    valid=valid&mutable
    allowed=gap>=-1e-4
    usable=valid&allowed.any(-1)
    near=(gap.clamp_min(0).square().sum(-1).sqrt()<.5)
    contact=((gap[...,0].abs()<1e-4)&(gap[...,1]<-1e-3))|((gap[...,1].abs()<1e-4)&(gap[...,0]<-1e-3))
    grouped=(g['group']>0)&valid
    return allowed,usable,near,contact,grouped

def objective(logits,g):
    allowed,valid,near,contact,grouped=targets(g)
    logprob=logits[...,:2].log_softmax(-1)
    nll=-torch.logsumexp(logprob.masked_fill(~allowed,-1e4),-1)
    weight=valid*(.25+.75*near)
    axis=(nll*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1)
    pos=grouped&contact;neg=grouped&~contact
    bce=F.binary_cross_entropy_with_logits(logits[...,2],contact.to(logits),reduction='none')
    positive=(bce*pos).sum((1,2))/pos.sum((1,2)).clamp_min(1)
    negative=(bce*neg).sum((1,2))/neg.sum((1,2)).clamp_min(1)
    edge=.5*(positive+negative)
    percase=axis+.5*edge
    best=logits[...,:2].argmax(-1)
    correct=allowed.gather(-1,best[...,None]).squeeze(-1)
    accuracy=(correct*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1)
    return {'per_case':percase,'axis_nll':axis,'contact_bce':edge,'axis_accuracy':accuracy}
