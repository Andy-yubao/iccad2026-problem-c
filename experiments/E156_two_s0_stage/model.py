"""Frozen E155 L0 followed by one shared two-layer S0 applied twice."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .refinement import PublicProblemEncoder,FunctionalUnit
from .public import PUBLIC
from .perturbation import perturb
class Model(nn.Module):
    unit_order=(0,0)
    def __init__(self,width=64,layers=4,rectangle=True,use_gap=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=2,phase='backend'):
        super().__init__();assert phase=='backend' and layers==4 and initial_layers==2
        self.phase=phase
        self.initial_generator=RoleInitial(width=width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
        self.initial_generator.requires_grad_(False).eval()
        self.problem_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
        self.s0=FunctionalUnit(width,ffn_dropout,residual_dropout,replan=True)
    def train(self,mode=True):
        super().train(mode)
        if self.phase=='backend':self.initial_generator.eval()
        return self
    def forward(self,g):
        if set(g)!=set(PUBLIC):raise ValueError('Public input contract mismatch')
        with torch.no_grad():
            clean=self.initial_generator(g)[0]
            state=perturb(clean,g) if self.training else clean
        states=[clean];condition=self.problem_encoder(g)
        for index in self.unit_order:
            output=self.s0(g,state,condition);states.extend(output.unbind(0));state=output[-1]
        return torch.stack(states)
def GeometryModel(architecture='e156',**configuration):
    if architecture!='e156':raise ValueError(architecture)
    return Model(**configuration)
