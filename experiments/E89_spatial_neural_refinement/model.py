"""Frozen public E90 placer plus one parallel, spatial neural residual head."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .placer_model import GeometryModel as Placer, decode


class GeometryModel(nn.Module):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,spatial_mode='spatial'):
        super().__init__()
        if spatial_mode not in ('spatial','pooled'):raise ValueError(spatial_mode)
        self.spatial_mode=spatial_mode
        self.placer=Placer(width=width,layers=layers,rectangle=rectangle,ffn_dropout=ffn_dropout)
        self.placer.requires_grad_(False);self.placer.eval()
        channels=64
        self.node=nn.Sequential(nn.Linear(17+8,channels),nn.SiLU(),nn.LayerNorm(channels))
        self.message=nn.Sequential(nn.Linear(channels*3,channels),nn.SiLU())
        self.grid_encoder=nn.Sequential(nn.Conv2d(channels+8,channels,3,padding=1),nn.GroupNorm(8,channels),nn.SiLU(),
            nn.Conv2d(channels,channels,3,padding=2,dilation=2),nn.GroupNorm(8,channels),nn.SiLU(),
            nn.Conv2d(channels,channels,3,padding=4,dilation=4),nn.SiLU())
        self.query=nn.Linear(channels,channels);self.key=nn.Linear(channels,channels)
        self.value=nn.Linear(channels,channels)
        self.crowding=nn.Parameter(torch.tensor(0.))
        self.competition=nn.Parameter(torch.tensor(0.))
        self.readout=nn.Sequential(nn.Linear(channels*2+2,128),nn.SiLU(),nn.Dropout(ffn_dropout))
        self.delta_head=nn.Linear(128,3)
        nn.init.zeros_(self.delta_head.weight);nn.init.zeros_(self.delta_head.bias)
        y,x=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
        self.register_buffer('grid_lo',torch.stack((x,y),-1).reshape(256,2).float()/16)

    def train(self,mode=True):
        super().train(mode);self.placer.eval();return self

    def spatial_state(self,g,source):
        valid=g['valid'];p=decode(source,g);lo=p[...,:2];hi=lo+p[...,2:]
        lower=lo.masked_fill(~valid[...,None],float('inf')).amin(1)
        upper=hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
        extent=(upper-lower).clamp_min(1e-6)
        a=(lo-lower[:,None])/extent[:,None];z=(hi-lower[:,None])/extent[:,None]
        overlap=(torch.minimum(z[:,:,None],self.grid_lo[None,None]+1/16)-
                 torch.maximum(a[:,:,None],self.grid_lo[None,None])).clamp_min(0)
        cover=(overlap[...,0]*overlap[...,1])*256*valid[:,:,None]
        preplaced=(~g['mask'][...,0].bool())&valid
        obstacle=(cover*preplaced[:,:,None]).sum(1)
        dynamic=(cover*(valid&~preplaced)[:,:,None]).sum(1)
        total=cover.sum(1)
        center=(a+z)/2;size=z-a
        local=g['area'].clamp_min(1e-12).sqrt()
        extras=torch.cat((center,size,source[...,2:3],local[...,None],g['mask'][...,:2]),-1)
        h=self.node(torch.cat((g['features'],extras),-1))*valid[...,None]
        h=(h+self.message(torch.cat((g['adjacency']@h,g['group']@h,g['mib']@h),-1)))*valid[...,None]
        field=cover.transpose(1,2)@h/total.clamp_min(1e-6)[...,None]
        population=torch.cat((field,dynamic[...,None],obstacle[...,None],(total-1).clamp_min(0)[...,None]),-1)
        others=(total[:,None]-cover).clamp_min(0)
        if self.spatial_mode=='pooled':
            population=population.mean(1,keepdim=True).expand_as(population)
            obstacle=obstacle.mean(1,keepdim=True).expand_as(obstacle)
            others=others.mean(-1,keepdim=True).expand_as(others)
        coords=(self.grid_lo+1/32)[None].expand(len(p),-1,-1)
        total_area=(g['area']*valid).sum(1).clamp_min(1e-12)
        frame=torch.cat((extent,((extent[:,0]*extent[:,1])/total_area)[:,None]),-1)
        # 64 learned fields +3 population +2 coordinates +3 frame channels.
        grid=torch.cat((population,coords,frame[:,None].expand(-1,256,-1)),-1)
        grid=self.grid_encoder(grid.transpose(1,2).reshape(len(p),72,16,16)).flatten(2).transpose(1,2)
        return h,grid,cover,others,obstacle,extent,valid,preplaced

    def forward(self,g):
        if set(g)-{'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the corrector')
        with torch.no_grad():source=self.placer(g)
        h,grid,cover,others,obstacle,extent,valid,preplaced=self.spatial_state(g,source)
        logits=(self.query(h)@self.key(grid).transpose(1,2))/8
        logits=logits-F.softplus(self.crowding)*others
        route=logits.softmax(-1)
        movable=(valid&~preplaced).to(h.dtype)
        area_fraction=g['area']/(extent[:,0]*extent[:,1])[:,None].clamp_min(1e-12)
        demand=(route*(area_fraction*movable)[:,:,None]).sum(1)
        capacity=(1-obstacle).clamp_min(1e-4)/256
        competition=torch.log1p(demand/capacity)
        route=(logits-F.softplus(self.competition)*competition[:,None]).softmax(-1)
        context=route@self.value(grid)
        crowd=(cover*others).sum(-1)/cover.sum(-1).clamp_min(1e-6)
        pressure=(route*competition[:,None]).sum(-1)
        hidden=self.readout(torch.cat((h,context,crowd[...,None],pressure[...,None]),-1))
        raw=self.delta_head(hidden).tanh()
        shift=raw[...,:2]*g['area'].clamp_min(1e-12).sqrt()[...,None]
        shape=.5*raw[...,2:3]
        return source+torch.cat((shift,shape),-1)*g['mask']*valid[...,None]
