"""Conditional shared vector field; no layout-label argument in public encoding."""
import math
import torch
from torch import nn

PUBLIC={'features','adjacency','group','constraints','valid','address','mask','area','known','specs','scale'}

def relation_descriptor(g):
    if set(g)-PUBLIC:raise ValueError('Label/unknown field in public condition')
    f,valid,address=g['features'],g['valid'],g['address']
    if f.shape[-1]!=16 or address.shape!=valid.shape or f.shape[1]>128:raise ValueError('Descriptor shape')
    active=valid[:,:,None]&valid[:,None,:]
    identity=(address[...,None]==torch.arange(128,device=f.device)).to(f.dtype)*valid[...,None]
    inverse=identity.argmax(1)[:,None,:].expand(-1,f.shape[1],-1)
    occupied=identity.sum(1)[:,None,:]
    wire=(g['adjacency']*active).gather(2,inverse)*occupied
    labels=g['constraints'][...,3]
    same=(labels[:,:,None]==labels[:,None,:])&(labels[:,:,None]>0)&active
    group=same.to(f.dtype).gather(2,inverse)*occupied
    return torch.cat((f,identity,wire,group),-1)*valid[...,None]

class FlowBlock(nn.Module):
    def __init__(self):
        super().__init__();d=96
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(d,4*d))
        self.qkv=nn.Linear(d,3*d);self.output=nn.Linear(d,d)
        self.ff_value=nn.Linear(d,192);self.ff_gate=nn.Linear(d,192);self.ff_output=nn.Linear(192,d)
        self.ff_dropout=nn.Dropout(.1);self.residual_dropout=nn.Dropout(.05)
    def forward(self,h,valid,temb):
        shift1,scale1,shift2,scale2=self.modulation(temb).chunk(4,-1)
        u=self.norm1(h)*(1+scale1[:,None])+shift1[:,None]
        b,n,_=h.shape
        q,k,v=self.qkv(u).view(b,n,3,4,24).permute(2,0,3,1,4).unbind(0)
        scores=((q@k.transpose(-2,-1))/math.sqrt(24)).masked_fill(~valid[:,None,None,:],-1e4)
        a=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,96)
        h=h+self.residual_dropout(self.output(a))
        u=self.norm2(h)*(1+scale2[:,None])+shift2[:,None]
        u=self.ff_value(u)*torch.nn.functional.silu(self.ff_gate(u))
        return (h+self.residual_dropout(self.ff_output(self.ff_dropout(u))))*valid[...,None]

class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(400,192),nn.SiLU(),nn.Linear(192,96),nn.LayerNorm(96))
        self.state_input=nn.Sequential(nn.Linear(3,96),nn.SiLU(),nn.Linear(96,96))
        self.time_input=nn.Sequential(nn.Linear(32,96),nn.SiLU(),nn.Linear(96,96))
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),16)))
        self.register_buffer('state_mean',torch.zeros(3));self.register_buffer('state_std',torch.ones(3))
        self.layers=nn.ModuleList(FlowBlock() for _ in range(6))
        self.head=nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,3))
    def standardize(self,state):return (state-self.state_mean)/self.state_std
    def unstandardize(self,z):return z*self.state_std+self.state_mean
    def encode(self,g):return self.input(relation_descriptor(g))*g['valid'][...,None]
    def velocity(self,condition,z,t,valid):
        phase=2*math.pi*t[:,None]*self.frequencies[None]
        temb=self.time_input(torch.cat((phase.sin(),phase.cos()),-1))
        h=(condition+self.state_input(z))*valid[...,None]
        for layer in self.layers:h=layer(h,valid,temb)
        context=h.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        return self.head(torch.cat((h,context[:,None].expand_as(h)),-1))*valid[...,None]
    def forward(self,g,z,t):return self.velocity(self.encode(g),z,t,g['valid'])

def build_model(arm='fm',seed=0):
    if arm!='fm':raise ValueError(arm)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return FlowModel()
