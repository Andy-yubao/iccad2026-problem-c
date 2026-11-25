"""Joint reusable two-layer operators; dynamic hidden never crosses unit calls."""
import math
import torch
from torch import nn
from .e119_model import InitialGraphBlock
from .original_model import RectangleBlock
from .geometry_feedback import module_features,pair_features
from .shape_feedback import shape_features

PUBLIC={'features','adjacency','group','mask','known','area','specs','constraints','valid','scale'}

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

    def forward(self,h,g,state,demand,pair):
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

class PublicProblemEncoder(nn.Module):
    """Pure immutable public problem function, independent of any layout/history."""
    def __init__(self,width,ffn_dropout,residual_dropout):
        super().__init__()
        self.input=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.LayerNorm(width))
        self.blocks=nn.ModuleList([InitialGraphBlock(width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout) for _ in range(2)])

    def forward(self,g):
        h=self.input(g['features'])*g['valid'][...,None]
        for layer in self.blocks:h=layer(h,g)
        return h

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

class FunctionalUnit(nn.Module):
    def __init__(self,width,ffn_dropout,residual_dropout,overlap_feedback=True):
        super().__init__()
        self.overlap_feedback=overlap_feedback
        self.layers=nn.ModuleList([CoordinatedBlock(width,ffn_dropout,residual_dropout) for _ in range(2)])
        self.demand_encoder=nn.ModuleList([nn.Sequential(nn.Linear(size,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width)) for size in (81,65)])
        self.state_embed=nn.ModuleList([nn.Linear(3,width) for _ in range(2)])
        self.heads=nn.ModuleList([ResidualHead(width) for _ in range(2)])

    def forward(self,g,state,problem_condition):
        h=problem_condition;states=[]
        for k in range(2):
            pair,p=pair_features(state,g,self.overlap_feedback)
            dynamic=torch.cat((module_features(p,g,self.overlap_feedback),
                               shape_features(p,g,self.overlap_feedback)),-1)
            # The second sublayer has no raw/static problem-conditioning path.
            demand_input=torch.cat((dynamic,g['features']),-1) if k==0 else dynamic
            demand=self.demand_encoder[k](demand_input)*g['valid'][...,None]
            h=self.layers[k](h+self.state_embed[k](state),g,state,demand,pair)
            state=self.heads[k](h,state,g);states.append(state)
        return torch.stack(states,0)

class SharedFunctionalModel(nn.Module):
    unit_order=(0,0,1,1,2,2)
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        super().__init__()
        if (layers,initial_layers,rectangle,use_gap)!=(12,2,True,True):raise ValueError('Unregistered architecture')
        self.input=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.LayerNorm(width))
        self.initial_blocks=nn.ModuleList([InitialGraphBlock(width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))
        self.problem_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
        self.units=nn.ModuleList([FunctionalUnit(width,ffn_dropout,residual_dropout) for _ in range(3)])

    def forward(self,g):
        if set(g)-PUBLIC:raise ValueError('Only public graph inputs may enter the network')
        h=self.input(g['features'])*g['valid'][...,None]
        for block in self.initial_blocks:h=block(h,g)
        state=self.head(h)*g['valid'][...,None]
        # Cache a pure P-only function. No initial/dynamic hidden enters this path.
        problem_condition=self.problem_encoder(g)
        states=[]
        for index in self.unit_order:
            emitted=self.units[index](g,state,problem_condition)
            states.extend(emitted.unbind(0));state=emitted[-1]
        return torch.stack(states,0)

def GeometryModel(architecture='e148',**configuration):
    if architecture!='e148':raise ValueError(architecture)
    return SharedFunctionalModel(**configuration)
