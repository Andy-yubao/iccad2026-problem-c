"""Sequential S1 then S2; frozen upstream stages always eval/no-grad."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .refinement import PublicProblemEncoder,FunctionalUnit
from .public import PUBLIC

class Model(nn.Module):
    def __init__(self,width=64,layers=8,rectangle=True,use_gap=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=2,phase='s1'):
        super().__init__();assert phase in ('s1','s2') and layers==({'s1':8,'s2':12}[phase]) and initial_layers==2
        self.phase=phase
        self.initial_generator=RoleInitial(width=width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
        self.s0_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
        self.s0=FunctionalUnit(width,ffn_dropout,residual_dropout,replan=True)
        self.s1_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
        self.s1=FunctionalUnit(width,ffn_dropout,residual_dropout,replan=False)
        if phase=='s2':
            self.s2_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
            self.s2=FunctionalUnit(width,ffn_dropout,residual_dropout,replan=False)
        self.frozen_names=('initial_generator','s0_encoder','s0')+(('s1_encoder','s1') if phase=='s2' else ())
        for name in self.frozen_names:getattr(self,name).requires_grad_(False).eval()
    def train(self,mode=True):
        super().train(mode)
        for name in self.frozen_names:getattr(self,name).eval()
        return self
    def apply_twice(self,unit,condition,state,g,states):
        for _ in range(2):
            out=unit(g,state,condition);states.extend(out.unbind(0));state=out[-1]
        return state
    def forward(self,g):
        if set(g)!=set(PUBLIC):raise ValueError('Public input contract mismatch')
        with torch.no_grad():
            state=self.initial_generator(g)[0];states=[state]
            state=self.apply_twice(self.s0,self.s0_encoder(g),state,g,states)
            if self.phase=='s2':state=self.apply_twice(self.s1,self.s1_encoder(g),state,g,states)
        if self.phase=='s1':state=self.apply_twice(self.s1,self.s1_encoder(g),state,g,states)
        else:state=self.apply_twice(self.s2,self.s2_encoder(g),state,g,states)
        return torch.stack(states)

def GeometryModel(architecture='e157',**configuration):
    if architecture!='e157':raise ValueError(architecture)
    return Model(**configuration)
