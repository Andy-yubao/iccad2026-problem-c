"""E9 attention geometry model; E7-compatible weights, no relation output heads.

GeometryBlock copied from retained E7 attention_geometry. The stopped rectangle
and occupancy routes are intentionally not included.
"""
import math
import torch
from torch import nn


class GeometryModel(nn.Module):
    def __init__(self, architecture='attention', width=96, layers=4):
        super().__init__()
        if architecture not in ('attention','center_refiner') or width % 4:
            raise ValueError('Unknown E9 geometry architecture or invalid width')
        self.architecture, self.width, self.depth = architecture, width, layers
        self.input = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList([GeometryBlock(width) for _ in range(layers)])
        self.state_embed = nn.Linear(3, width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        if architecture=='center_refiner':
            # Freeze the complete retained model, including its shape decoder.
            for parameter in self.parameters():parameter.requires_grad_(False)
            self.center_refiner=nn.Sequential(nn.Linear(width+28,width),nn.SiLU(),nn.Linear(width,2))
            nn.init.zeros_(self.center_refiner[-1].weight)
            nn.init.zeros_(self.center_refiner[-1].bias)

    @property
    def config(self):
        return {'architecture': self.architecture, 'width': self.width, 'layers': self.depth}

    def forward(self, g):
        if 'target' in g or 'reference' in g:
            raise ValueError('Supervision must not enter the geometry network')
        h = self.input(g['features']) * g['valid'][..., None]
        state = self.head(h)
        for layer in self.layers:
            h = layer(h + self.state_embed(state * g['mask']), g, state)
            state = self.head(h)
        if self.architecture=='center_refiner':
            public_state=state*g['mask']+g['known']
            xy=public_state[...,:2]
            messages=[torch.bmm(g[key],xy)-g[key].sum(-1,keepdim=True)*xy for key in ('adjacency','mib','group')]
            mean=(xy*g['valid'][...,None]).sum(1,keepdim=True)/g['valid'].sum(1)[:,None,None]
            messages.append(mean.expand_as(xy)-xy)
            features=torch.cat((h,public_state,g['features'],*messages),-1)
            delta=self.center_refiner(features)*g['area'].clamp_min(1e-12).sqrt()[...,None]*g['mask'][...,:2]
            # Only centers can change; frozen log ratios remain exactly equal.
            state=torch.cat((state[...,:2]+delta,state[...,2:]),-1)
        return state


class GeometryBlock(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__()
        self.heads=heads;self.dim=width//heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3);self.output=nn.Linear(width,width)
        self.local=nn.Linear(width*3,width)
        self.bias=nn.Sequential(nn.Linear(7,24),nn.SiLU(),nn.Linear(24,heads))
        self.ff=nn.Sequential(nn.Linear(width,width*3),nn.SiLU(),nn.Linear(width*3,width))

    def forward(self,h,g,state):
        b,n,w=h.shape
        q,k,v=self.qkv(self.norm1(h)).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        # Internal pair features are neither model outputs nor relation heads.
        xy=torch.where(g['mask'][...,:2].bool(),state[...,:2],g['known'][...,:2])
        delta=xy[:,:,None,:]-xy[:,None,:,:]
        pair=torch.cat((delta,delta.abs(),g['adjacency'][...,None],
                        (g['mib']>0)[...,None],(g['group']>0)[...,None]),dim=-1)
        bias=self.bias(pair).permute(0,3,1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)+bias
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,w)
        local=self.local(torch.cat((torch.bmm(g['adjacency'],h),torch.bmm(g['mib'],h),torch.bmm(g['group'],h)),dim=-1))
        h=h+self.output(attended)+local
        return (h+self.ff(self.norm2(h)))*g['valid'][...,None]
