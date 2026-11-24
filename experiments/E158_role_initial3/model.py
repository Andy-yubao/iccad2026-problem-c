"""L0 only; no intermediate coordinates, geometry feedback or backend."""
from torch import nn
from .initial_model import RoleInitial

class Model(nn.Module):
    def __init__(self,width=64,layers=3,rectangle=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=3):
        super().__init__()
        self.initial_generator=RoleInitial(width=width,layers=layers,rectangle=rectangle,
            ffn_dropout=ffn_dropout,residual_dropout=residual_dropout,initial_layers=initial_layers)

    def forward(self,g):
        return self.initial_generator(g)

def GeometryModel(architecture='e158',**configuration):
    if architecture!='e158':raise ValueError('Only E158 L0 is registered')
    return Model(**configuration)
