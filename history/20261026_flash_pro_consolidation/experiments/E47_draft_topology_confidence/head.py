"""Frozen-draft auxiliary relation probe. Inputs contain public features only."""
import numpy as np
import torch
from torch import nn
from .inference import graph
from .model import decode


class RelationHead(nn.Module):
    def __init__(self,width=96):
        super().__init__();self.node=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,32),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,4))

    def forward(self,g):
        if set(g)!={'latent','draft','adjacency','constraints'}:raise ValueError('Auxiliary inputs must be public')
        z=g['draft'];b,n,_=z.shape;i,j=torch.tril_indices(n,n,-1,device=z.device)
        h=self.node(g['latent']);wh=z[...,2:];center=z[...,:2]+wh/2
        delta=center[:,i]-center[:,j];wi,wj=wh[:,i],wh[:,j]
        ai,aj=wi.prod(-1).clamp_min(1e-10),wj.prod(-1).clamp_min(1e-10)
        unit=((ai.sqrt()+aj.sqrt())/2).clamp_min(1e-6);c=g['constraints']
        same=[((c[:,i,col]==c[:,j,col])&(c[:,i,col]>0)).to(z.dtype)[...,None] for col in (2,3)]
        geom=torch.cat((delta,delta.abs(),wi,wj,((delta.abs()-(wi+wj)/2)/unit[...,None]).clamp(-20,20),
            (ai/aj).log()[...,None].clamp(-10,10),unit[...,None],g['adjacency'][:,i,j,None],g['adjacency'][:,j,i,None],
            *same,c[:,i,:2].to(z.dtype),c[:,j,:2].to(z.dtype)),-1)
        return self.pair(torch.cat((h[:,i]+h[:,j],h[:,i]-h[:,j],geom),-1))


def collate(rows,device):
    b=len(rows);n=max(r['n'] for r in rows);k=n*(n-1)//2
    g={'latent':torch.zeros(b,n,96),'draft':torch.zeros(b,n,4),'adjacency':torch.zeros(b,n,n),'constraints':torch.zeros(b,n,5,dtype=torch.long)}
    y=torch.zeros(b,k,4);mask=torch.zeros(b,k)
    for at,r in enumerate(rows):
        m=r['n'];count=m*(m-1)//2
        for key in ('latent','draft','constraints'):g[key][at,:m]=r[key]
        g['adjacency'][at,:m,:m]=r['adjacency'];y[at,:count]=r['labels'];mask[at,:count]=1/count
    return {key:value.to(device) for key,value in g.items()},y.to(device),mask.to(device)


def loss(logits,y,mask):
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none').mean(-1)*mask).sum()/len(y)


def labels(reference):
    i,j=torch.tril_indices(len(reference),len(reference),-1);r=reference.double();eps=1e-6
    return torch.stack((r[i,0]+r[i,2]<=r[j,0]+eps,r[j,0]+r[j,2]<=r[i,0]+eps,
        r[i,1]+r[i,3]<=r[j,1]+eps,r[j,1]+r[j,3]<=r[i,1]+eps),-1).to(torch.uint8)


@torch.no_grad()
def predict_latent(model,inputs):
    import time
    started=time.perf_counter();holder={}
    handle=model.head.register_forward_pre_hook(lambda module,args:holder.__setitem__('latent',args[0].detach()))
    try:g=graph(inputs);positions=(decode(model(g),g)[0]*g['scale'][0]).numpy().astype(float);latent=holder['latent'].clone()
    finally:handle.remove()
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    positions[locked,2:]=s[locked,2:];positions[c[:,1]!=0]=s[c[:,1]!=0]
    features={'latent':latent,'draft':torch.as_tensor(positions/g['scale'][0].item(),dtype=torch.float32)[None],
        'adjacency':g['adjacency'],'constraints':g['constraints']}
    return positions,features,time.perf_counter()-started
