"""Joint reusable two-layer operators; dynamic hidden never crosses unit calls."""
import math
import torch
from torch import nn
from .e119_model import InitialGraphBlock
from .original_model import RectangleBlock
from .geometry_feedback import module_features,pair_features
from .shape_feedback import shape_features
from .demand import extra_features
from .initial_model import RoleInitial

PUBLIC={'features','adjacency','group','mask','known','area','specs','constraints','valid','scale'}

class CoordinatedBlock(nn.Module):
    def __init__(self,width=64,ffn_dropout=.1,residual_dropout=.05,relation_width=8):
        super().__init__()
        base=RectangleBlock(width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
        for name in ('norm1','norm2','qkv','output','local','bias','ff','ff_dropout','residual_dropout'):
            setattr(self,name,getattr(base,name))
        self.heads,self.dim=base.heads,base.dim
        self.fuse=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width))
        self.bias=nn.Sequential(nn.Linear(21,24),nn.SiLU(),nn.Linear(24,self.heads))
        self.edge=nn.Sequential(nn.Linear(21,2*relation_width),nn.SiLU(),nn.Linear(2*relation_width,2*relation_width))
        self.mib_local=nn.Linear(width,width,bias=False)
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
        local=self.local(torch.cat((g['adjacency']@t,g['group']@t),-1))+self.mib_local(g['mib']@t)
        t=t+self.residual_dropout(self.output(attended)+local)
        ff=self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](self.norm2(t)))))
        return (t+self.residual_dropout(ff))*g['valid'][...,None]

class PublicProblemEncoder(nn.Module):
    """Pure immutable public problem function, independent of any layout/history."""
    def __init__(self,width,ffn_dropout,residual_dropout):
        super().__init__()
        self.encoder=RoleInitial(width=width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
        del self.encoder.head

    def forward(self,g):
        return self.encoder.encode(g)

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
    def __init__(self,width,ffn_dropout,residual_dropout,replan=False):
        super().__init__()
        self.layers=nn.ModuleList([CoordinatedBlock(width,ffn_dropout,residual_dropout) for _ in range(2)])
        self.demand_encoder=nn.ModuleList([nn.Sequential(nn.Linear(92,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width)) for _ in range(2)])
        self.state_embed=nn.ModuleList([nn.Linear(3,width) for _ in range(2)])
        self.heads=nn.ModuleList([ResidualHead(width) for _ in range(2)])
        self.replan=replan
        if replan:
            self.group_move=nn.ModuleList([nn.Linear(width,2) for _ in range(2)])
            self.member_move=nn.ModuleList([nn.Linear(2*width,2) for _ in range(2)])
            self.move_gate=nn.ModuleList([nn.Linear(2*width,1) for _ in range(2)])
            with torch.no_grad():
                for layer in (*self.group_move,*self.member_move):layer.weight.mul_(.01);layer.bias.zero_()
                for layer in self.move_gate:layer.weight.zero_();layer.bias.fill_(-2.)

    def forward(self,g,state,problem_condition):
        h=problem_condition;states=[]
        for k in range(2):
            pair,p=pair_features(state,g)
            dynamic=torch.cat((module_features(p,g),shape_features(p,g),extra_features(p,g)),-1)
            # The second sublayer has no raw/static problem-conditioning path.
            demand_input=torch.cat((dynamic,g['features']),-1)
            demand=self.demand_encoder[k](demand_input)*g['valid'][...,None]
            h=self.layers[k](h+self.state_embed[k](state)+(problem_condition if k else 0),g,state,demand,pair)
            state=self.heads[k](h,state,g)
            if self.replan:
                n=h.shape[1];eye=torch.eye(n,device=h.device,dtype=h.dtype)[None]
                group=(g['group']>0).to(h.dtype)+eye
                group=group*g['valid'][:,None,:];group=group/group.sum(-1,keepdim=True).clamp_min(1)
                pooled=group@h;context=torch.cat((h,pooled),-1)
                group_area=(group@g['area'][...,None]).clamp_min(1e-12).sqrt()
                move=(self.group_move[k](pooled).tanh()+self.member_move[k](context).tanh())*group_area
                shift=self.move_gate[k](context).sigmoid()*move
                state=torch.cat((state[...,:2]+shift,state[...,2:]),-1)*g['valid'][...,None]
            states.append(state)
        return torch.stack(states,0)

