"""Two public-graph global interactions before S0, then unchanged E114 refinement."""
import math
import torch
from torch import nn
from .original_model import GeometryModel as OriginalModel
from .original_model import decode

class InitialGraphBlock(nn.Module):
    """Geometry-free pre-norm attention and explicit wire/group graph messages."""
    def __init__(self,width=64,heads=4,ffn_dropout=.1,residual_dropout=.05):
        super().__init__()
        self.heads=heads;self.dim=width//heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3);self.output=nn.Linear(width,width)
        self.local=nn.Linear(width*2,width)
        self.ff=nn.Sequential(nn.Linear(width,width*3),nn.SiLU(),nn.Linear(width*3,width))
        self.ff_dropout=nn.Dropout(ffn_dropout);self.residual_dropout=nn.Dropout(residual_dropout)

    def forward(self,h,g):
        b,n,w=h.shape;normalized=self.norm1(h)
        q,k,v=self.qkv(normalized).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,w)
        local=self.local(torch.cat((g['adjacency']@normalized,g['group']@normalized),-1))
        h=h+self.residual_dropout(self.output(attended)+local)
        ff=self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](self.norm2(h)))))
        return (h+self.residual_dropout(ff))*g['valid'][...,None]

class GeometryModel(OriginalModel):
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,
                 ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        if initial_layers!=2:raise ValueError('Only the registered two initial graph blocks are authorized')
        super().__init__(width,layers,rectangle,use_gap,ffn_dropout,residual_dropout)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261109119)
            self.initial_blocks=nn.ModuleList([InitialGraphBlock(width,ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout) for _ in range(initial_layers)])

    def forward(self,g):
        if set(g)-{'features','adjacency','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        h=self.input(g['features'])*g['valid'][...,None]
        for block in self.initial_blocks:h=block(h,g)
        state=self.head(h)
        states=[state]  # Observe the existing S0; no extra head call.
        for layer in self.layers:
            h=layer(h+self.state_embed(state*g['mask']),g,state)
            state=self.head(h);states.append(state)
        return torch.stack(states,dim=0)
