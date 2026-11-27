"""First-block geometry attention and the stage residual head from E169."""
import math
import torch
from torch import nn
from .original_model import RectangleBlock

class CoordinatedBlock(nn.Module):
    def __init__(self,width=64,ffn_dropout=.1,residual_dropout=.05,relation_width=8):
        super().__init__()
        base=RectangleBlock(width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
        for name in ('norm1','norm2','qkv','output','local','bias','ff','ff_dropout','residual_dropout'):
            setattr(self,name,getattr(base,name))
        self.heads,self.dim=base.heads,base.dim
        self.fuse=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width))
        self.edge=nn.Sequential(nn.Linear(18,2*relation_width),nn.SiLU(),nn.Linear(2*relation_width,2*relation_width))
        self.receiver=nn.Linear(width,relation_width,bias=False)
        self.sender=nn.Linear(width,relation_width,bias=False)
        self.relation_value=nn.ModuleList([nn.Linear(relation_width,self.dim,bias=False) for _ in range(self.heads)])

    def forward(self,h,g,demand,pair):
        b,n,w=h.shape
        t=h+self.fuse(torch.cat((h,demand),-1));normalized=self.norm1(t)
        q,k,v=self.qkv(normalized).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)+self.bias(pair).permute(0,3,1,2)
        attention=scores.masked_fill(~g['valid'][:,None,None,:],-1e4).softmax(-1)
        content,gate=self.edge(pair).chunk(2,-1)
        relation=torch.tanh(content+self.receiver(normalized)[:,:,None,:]+self.sender(normalized)[:,None,:,:])*gate.sigmoid()
        aggregate=torch.einsum('bhij,bijd->bhid',attention,relation)
        messages=torch.stack([projection(aggregate[:,a]) for a,projection in enumerate(self.relation_value)],1)
        attended=((attention@v)+messages).transpose(1,2).reshape(b,n,w)
        local=self.local(torch.cat((g['adjacency']@t,g['group']@t),-1))
        t=t+self.residual_dropout(self.output(attended)+local)
        ff=self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](self.norm2(t)))))
        return (t+self.residual_dropout(ff))*g['valid'][...,None]

class ResidualHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(width,width),nn.SiLU())
        self.position=nn.Linear(width,2);self.shape=nn.Linear(width,1)
        with torch.no_grad():
            for layer in (self.position,self.shape):layer.weight.mul_(.01);layer.bias.zero_()

    def forward(self,h,state,g):
        z=self.trunk(h);local=g['area'].clamp_min(1e-12).sqrt()[...,None]
        delta=torch.cat((self.position(z)*local,self.shape(z)),-1)
        return (state+delta)*g['valid'][...,None]

