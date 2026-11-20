"""One scratch design: role/MIB L0, enhanced conditions everywhere, S0 replanning."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .refinement import PublicProblemEncoder,FunctionalUnit
from .public import PUBLIC
class Model(nn.Module):
    unit_order=(0,0,1,1,2,2)
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=2,phase='backend'):
        super().__init__();assert phase in ('initial','backend') and layers==12 and initial_layers==2
        self.phase=phase
        self.initial_generator=RoleInitial(width=width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
        if phase=='backend':
            self.initial_generator.requires_grad_(False).eval()
            self.problem_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
            self.units=nn.ModuleList([FunctionalUnit(width,ffn_dropout,residual_dropout,replan=k==0) for k in range(3)])
    def train(self,mode=True):
        super().train(mode)
        if self.phase=='backend':self.initial_generator.eval()
        return self
    def forward(self,g):
        if set(g)!=set(PUBLIC):raise ValueError('Public input contract mismatch')
        if self.phase=='initial':return self.initial_generator(g)
        with torch.no_grad():state=self.initial_generator(g)[0]
        states=[state];condition=self.problem_encoder(g)
        for index in self.unit_order:
            output=self.units[index](g,state,condition);states.extend(output.unbind(0));state=output[-1]
        return torch.stack(states)
def GeometryModel(architecture='e154',**configuration):
    if architecture!='e154':raise ValueError(architecture)
    return Model(**configuration)
