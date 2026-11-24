"""Four independent stages; hidden-only blocks, one layout exit per stage."""
import torch
from torch import nn
from .initial_model import RoleInitial, RoleBlock
from .public import PUBLIC


def conditions(g):
    n=g['valid'].shape[1];edges=g['role_edges'];hops=g['role_hops'];t=edges.shape[2]
    ids=torch.arange(t,device=edges.device)
    eye=(ids[:n,None]==ids[None,:]).to(edges.dtype)[None].expand(edges.shape[0],-1,-1)
    pair=torch.stack((edges.log1p(),(edges>0).to(edges.dtype),(g['role_group']>0).to(edges.dtype),
        hops.clamp_min(0).to(edges.dtype).asinh(),(hops>=0).to(edges.dtype),eye),-1)
    return pair,edges/edges.sum(-1,keepdim=True).clamp_min(1e-6),torch.cat((g['valid'],g['port_valid']),1)


class Refinement(nn.Module):
    def __init__(self,width,ffn_dropout,residual_dropout):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.LayerNorm(width))
        self.role_input=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.LayerNorm(width))
        self.port_input=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.LayerNorm(width))
        self.layout_input=nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.LayerNorm(width))
        self.blocks=nn.ModuleList([RoleBlock(width,4,ffn_dropout,residual_dropout) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))
        with torch.no_grad():
            self.head[-1].weight.mul_(.01);self.head[-1].bias.zero_()

    def forward(self,g,state):
        local=g['area'].clamp_min(1e-12).sqrt()[...,None]
        state_features=torch.cat((state[...,:2]/local,state[...,2:]),-1)
        h=(self.input(g['features'])+self.role_input(g['role_features'])+self.layout_input(state_features))*g['valid'][...,None]
        ports=self.port_input(g['port_features'])*g['port_valid'][...,None]
        pair,adjacency,valid=conditions(g)
        for block in self.blocks:
            h=block(h,ports,valid,pair,adjacency,g['role_group'])
        delta=self.head(h)
        delta=torch.cat((delta[...,:2]*local,delta[...,2:]),-1)
        return (state+delta)*g['valid'][...,None]


class Model(nn.Module):
    def __init__(self,width=128,layers=3,rectangle=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=3,active_stage=0):
        super().__init__()
        if width!=128 or layers!=3 or initial_layers!=3 or active_stage not in range(4):
            raise ValueError('Unregistered E161 architecture')
        self.active_stage=active_stage
        self.initial_generator=RoleInitial(width,layers,rectangle,ffn_dropout,residual_dropout,initial_layers)
        self.stages=nn.ModuleList([Refinement(width,ffn_dropout,residual_dropout) for _ in range(active_stage)])
        self.requires_grad_(False)
        (self.initial_generator if active_stage==0 else self.stages[-1]).requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        if self.active_stage:
            self.initial_generator.eval()
            for stage in self.stages[:-1]:stage.eval()
        return self

    def forward(self,g):
        if set(g)!=set(PUBLIC):raise ValueError('Public-only no-MIB inputs required')
        if self.active_stage==0:return self.initial_generator(g)
        with torch.no_grad():
            state=self.initial_generator(g)[0];states=[state]
            for stage in self.stages[:-1]:
                state=stage(g,state);states.append(state)
        state=self.stages[-1](g,state);states.append(state)
        return torch.stack(states)


def GeometryModel(architecture='e161',**configuration):
    if architecture!='e161':raise ValueError(architecture)
    return Model(**configuration)
