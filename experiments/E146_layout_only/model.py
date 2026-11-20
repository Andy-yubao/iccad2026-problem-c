"""Local demand -> QKV communication + pair bias and pair-valued messages."""
import math
import torch
from torch import nn
from .e119_model import GeometryModel as OriginalE119
from .geometry_feedback import module_features,pair_features


class CoordinatedBlock(nn.Module):
    def __init__(self,base,width=64,relation_width=8):
        super().__init__()
        for name in ('norm1','norm2','qkv','output','local','bias','ff','ff_dropout','residual_dropout'):
            setattr(self,name,getattr(base,name))
        self.heads,self.dim=base.heads,base.dim
        self.fuse=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width))
        self.edge=nn.Sequential(nn.Linear(18,2*relation_width),nn.SiLU(),nn.Linear(2*relation_width,2*relation_width))
        self.receiver=nn.Linear(width,relation_width,bias=False)
        self.sender=nn.Linear(width,relation_width,bias=False)
        self.relation_value=nn.ModuleList([nn.Linear(relation_width,self.dim,bias=False) for _ in range(self.heads)])

    def forward(self,h,g,state,demand,pair):
        b,n,w=h.shape
        # Preserve own context/needs outside the attention aggregation.
        t=h+self.fuse(torch.cat((h,demand),-1))
        normalized=self.norm1(t)
        q,k,v=self.qkv(normalized).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)+self.bias(pair).permute(0,3,1,2)
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attention=scores.softmax(-1)
        content,gate=self.edge(pair).chunk(2,-1)
        relation=torch.tanh(content+self.receiver(normalized)[:,:,None,:]+self.sender(normalized)[:,None,:,:])*gate.sigmoid()
        # Share the8-dimensional edge basis across heads. Project only AFTER
        # reducing over senders; no BN²x64 value tensor is materialized.
        aggregate=torch.einsum('bhij,bijd->bhid',attention,relation)
        messages=torch.stack([projection(aggregate[:,a]) for a,projection in enumerate(self.relation_value)],1)
        attended=((attention@v)+messages).transpose(1,2).reshape(b,n,w)
        local=self.local(torch.cat((g['adjacency']@t,g['group']@t),-1))
        t=t+self.residual_dropout(self.output(attended)+local)
        ff=self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](self.norm2(t)))))
        return (t+self.residual_dropout(ff))*g['valid'][...,None]


class DualGeometryModel(OriginalE119):
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        if not rectangle or not use_gap:raise ValueError('Registered geometry feedback requires original rectangle/gap channels')
        super().__init__(width,layers,rectangle,use_gap,ffn_dropout,residual_dropout,initial_layers)
        self.demand_encoder=nn.Sequential(nn.Linear(49+16,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width))
        self.layers=nn.ModuleList([CoordinatedBlock(base,width) for base in self.layers])

    def forward(self,g):
        if set(g)-{'features','adjacency','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        static=self.input(g['features'])*g['valid'][...,None]
        h=static
        for block in self.initial_blocks:h=block(h,g)
        state=self.head(h);states=[]
        for layer in self.layers:
            pair,p=pair_features(state,g)
            dynamic=module_features(p,g)
            demand=self.demand_encoder(torch.cat((dynamic,g['features']),-1))*g['valid'][...,None]
            h=layer(static+self.state_embed(state*g['mask']),g,state,demand,pair)
            state=self.head(h);states.append(state)
        return torch.stack(states,0)


def GeometryModel(architecture='e146',**configuration):
    if architecture!='e146':raise ValueError(architecture)
    return DualGeometryModel(**configuration)
