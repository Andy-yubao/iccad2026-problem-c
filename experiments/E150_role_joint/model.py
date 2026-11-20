"""Role-aware initial generation and unchanged12 geometry updates, jointly trainable."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .refinement import PublicProblemEncoder,FunctionalUnit
from .public import PUBLIC,BASE

class JointRoleModel(nn.Module):
    unit_order=(0,0,1,1,2,2)
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        super().__init__()
        if (layers,initial_layers,rectangle,use_gap)!=(12,2,True,True):raise ValueError('Unregistered architecture')
        self.initial_generator=RoleInitial(width=width,layers=2,rectangle=True,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout,initial_layers=2)
        self.problem_encoder=PublicProblemEncoder(width,ffn_dropout,residual_dropout)
        self.units=nn.ModuleList([FunctionalUnit(width,ffn_dropout,residual_dropout) for _ in range(3)])

    def forward(self,g):
        if set(g)!=set(PUBLIC):raise ValueError('Only registered public inputs accepted')
        state=self.initial_generator(g)[0]
        public={key:g[key] for key in BASE}
        condition=self.problem_encoder(public)
        states=[state]
        for index in self.unit_order:
            output=self.units[index](public,state,condition)
            states.extend(output.unbind(0));state=output[-1]
        return torch.stack(states,0)

def GeometryModel(architecture='e150',**configuration):
    if architecture!='e150':raise ValueError(architecture)
    return JointRoleModel(**configuration)
