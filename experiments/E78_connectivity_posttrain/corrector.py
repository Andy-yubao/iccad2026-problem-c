"""Public geometry pair messages; only a bounded local correction is learned."""
import torch
from torch import nn
from .base_model import decode

class Corrector(nn.Module):
    def __init__(self,width=96,message_width=32,dropout=.05):
        super().__init__()
        self.node=nn.Sequential(nn.Linear(22,width),nn.SiLU(),nn.LayerNorm(width))
        self.receiver=nn.Linear(width,message_width)
        self.sender=nn.Linear(width,message_width,bias=False)
        self.edge=nn.Linear(17,message_width)
        self.score=nn.Linear(message_width,1)
        self.output=nn.Sequential(nn.Linear(width+message_width,width),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width,3))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,state,g):
        p=decode(state,g);local=g['area'].clamp_min(1e-12).sqrt();center=p[...,:2]+p[...,2:]/2
        shape=(p[...,2:]/local[...,None]).clamp_min(1e-6).log().clamp(-5,5)
        h=self.node(torch.cat((g['features'],state,shape),-1))*g['valid'][...,None]
        size=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
        delta=(center[:,:,None]-center[:,None,:])/size[...,None]
        gap=delta.abs()-(p[:,:,None,2:]+p[:,None,:,2:])/(2*size[...,None])
        n=state.shape[1];flags=g['constraints'][...,:2].to(state.dtype)
        edge=torch.cat((delta.clamp(-8,8),gap.clamp(-8,8),(-gap).clamp(0,8),
            shape[:,:,None].expand(-1,-1,n,-1),shape[:,None,:].expand(-1,n,-1,-1),
            torch.stack((g['adjacency'],(g['mib']>0).to(state.dtype),(g['group']>0).to(state.dtype)),-1),
            flags[:,:,None].expand(-1,-1,n,-1),flags[:,None,:].expand(-1,n,-1,-1)),-1)
        value=torch.nn.functional.silu(self.receiver(h)[:,:,None]+self.sender(h)[:,None,:]+self.edge(edge))
        valid=g['valid'][:,:,None]&g['valid'][:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
        near=(gap.clamp_min(0).sum(-1)<=.5)|(g['adjacency']>0)|(g['group']>0)|(g['mib']>0)
        valid=valid&near
        scores=self.score(value).squeeze(-1).masked_fill(~valid,-1e4)
        attention=scores.softmax(-1)*valid
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        message=(value*attention[...,None]).sum(2)
        return self.output(torch.cat((h,message),-1))*g['valid'][...,None]
