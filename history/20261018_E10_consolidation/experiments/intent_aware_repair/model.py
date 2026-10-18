"""Geometry-only batched baseline and relative-geometry attention.

Baseline equations copied from E1 GraphEncoder on 2026-10-17, vectorized here.
Every architecture returns exactly B x N x 3 center/log-ratio values.
"""
import math
import torch
from torch import nn


class GeometryModel(nn.Module):
    def __init__(self, architecture='baseline', width=None, layers=None):
        super().__init__()
        self.architecture=architecture
        self.width=width or (64 if architecture=='baseline' else 96)
        self.depth=layers or (3 if architecture=='baseline' else 4)
        w=self.width
        self.input=nn.Sequential(nn.Linear(17,w),nn.SiLU(),nn.LayerNorm(w))
        if architecture=='baseline':
            self.layers=nn.ModuleList([nn.Sequential(nn.Linear(w*5,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(self.depth)])
            self.norms=nn.ModuleList([nn.LayerNorm(w) for _ in range(self.depth)])
        elif architecture=='attention':
            self.layers=nn.ModuleList([GeometryBlock(w) for _ in range(self.depth)])
            self.state_embed=nn.Linear(3,w)
        else:
            raise ValueError(architecture)
        self.head=nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,3))

    @property
    def config(self):
        return {'architecture':self.architecture,'width':self.width,'layers':self.depth}

    def forward(self,g):
        if 'target' in g or 'reference' in g:
            raise ValueError('Supervision must not be passed into the geometry network')
        h=self.input(g['features'])*g['valid'][...,None]
        if self.architecture=='baseline':
            for layer,norm in zip(self.layers,self.norms):
                summary=(h*g['valid'][...,None]).sum(1,keepdim=True)/g['valid'].sum(1)[:,None,None]
                x=torch.cat((h,torch.bmm(g['adjacency'],h),torch.bmm(g['mib'],h),
                             torch.bmm(g['group'],h),summary.expand_as(h)),dim=-1)
                h=norm(h+layer(x))*g['valid'][...,None]
            return self.head(h)
        state=self.head(h)
        for layer in self.layers:
            h=layer(h+self.state_embed(state*g['mask']),g,state)
            state=self.head(h)
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
