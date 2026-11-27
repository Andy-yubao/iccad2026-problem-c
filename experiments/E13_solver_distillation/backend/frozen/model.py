"""Geometry-only batched baseline and relative-geometry attention.

Baseline equations copied from E1 GraphEncoder on 2026-10-17, vectorized here.
Every architecture returns exactly B x N x 3 center/log-ratio values.
"""
import math
import torch
from torch import nn
from .batches import decode


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
        elif architecture in ('attention','rectangle'):
            block = GeometryBlock if architecture=='attention' else RectangleBlock
            self.layers=nn.ModuleList([block(w) for _ in range(self.depth)])
            self.state_embed=nn.Linear(3,w)
            if architecture=='rectangle':
                self.update=nn.Linear(w,3)
                nn.init.zeros_(self.update.weight)
                nn.init.zeros_(self.update.bias)
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
            if self.architecture=='rectangle':
                local=g['area'].clamp_min(1e-12).sqrt()
                step_scale=torch.stack((local,local,torch.full_like(local,.5)),-1)
                state=state+step_scale*torch.tanh(self.update(h))
            else:
                state=self.head(h)
        return state


class RectangleBlock(nn.Module):
    """Current rectangle geometry conditions attention and a collision message.

    Pair features remain internal; the only output is the updated node embedding.
    Unlike the E6 center-only bias, widths, gaps and intersections are explicit.
    """
    def __init__(self,width,heads=4):
        super().__init__()
        if width%heads:raise ValueError('Width must be divisible by heads')
        self.heads=heads;self.dim=width//heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.output=nn.Linear(width,width)
        self.bias=nn.Sequential(nn.Linear(11,24),nn.SiLU(),nn.Linear(24,heads))
        self.collision=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width))
        self.local=nn.Linear(3*width,width)
        self.ff=nn.Sequential(nn.Linear(width,3*width),nn.SiLU(),nn.Linear(3*width,width))

    def forward(self,h,g,state):
        b,n,w=h.shape
        p=decode(state,g);wh=p[...,2:].clamp_min(1e-8);center=p[...,:2]+wh/2
        delta=center[:,:,None]-center[:,None,:]
        extent=(wh[:,:,None]+wh[:,None,:])/2
        gap=delta.abs()-extent
        intersection=(torch.minimum((p[...,:2]+wh)[:,:,None],(p[...,:2]+wh)[:,None,:])-
                      torch.maximum(p[...,:2][:,:,None],p[...,:2][:,None,:])).clamp_min(0)
        geom=torch.cat((delta/extent,gap/extent,intersection/extent,
                        (wh[:,:,None]/wh[:,None,:]).log()),-1).clamp(-12,12)
        pair=torch.cat((geom,g['adjacency'][...,None],(g['mib']>0)[...,None],
                        (g['group']>0)[...,None]),-1)
        q,k,v=self.qkv(self.norm1(h)).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)+self.bias(pair).permute(0,3,1,2)
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,w)
        valid_pairs=g['valid'][:,:,None]&g['valid'][:,None,:]&~torch.eye(n,device=h.device,dtype=torch.bool)[None]
        weight=(intersection.prod(-1)/torch.minimum(wh.prod(-1)[:,:,None],wh.prod(-1)[:,None,:]).clamp_min(1e-8)).clamp(0,1)*valid_pairs
        degree=weight.sum(-1)
        neighborhood=(geom*weight[...,None]).sum(2)/degree.clamp_min(1)[...,None]
        collision=self.collision(torch.cat((neighborhood,degree.log1p()[...,None]),-1))
        local=self.local(torch.cat((g['adjacency']@h,g['mib']@h,g['group']@h),-1))
        h=h+self.output(attended)+local+collision
        return (h+self.ff(self.norm2(h)))*g['valid'][...,None]


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
