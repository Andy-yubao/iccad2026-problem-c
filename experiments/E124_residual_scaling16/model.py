"""S-state residual updates; original E119 readout generates S0 only."""
import torch
from torch import nn
from .reference_model import GeometryModel as E119Model
from .original_model import decode

class GeometryModel(E119Model):
    def __init__(self,width=64,layers=16,rectangle=True,use_gap=True,
                 ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        super().__init__(width,layers,rectangle,use_gap,ffn_dropout,residual_dropout,initial_layers)
        # New small per-block output layers, independent of the preserved S0 head.
        # Small nonzero weights keep the untrained state recursion near identity,
        # without the zero-norm lock under the existing relative-step guard.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261109121)
            self.delta_heads=nn.ModuleList([nn.Linear(width,3) for _ in range(layers)])
        with torch.no_grad():
            for head in self.delta_heads:
                head.weight.mul_(.01)
                head.bias.mul_(.01)

    def forward(self,g):
        if set(g)-{'features','adjacency','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        h=self.input(g['features'])*g['valid'][...,None]
        for block in self.initial_blocks:h=block(h,g)
        state=self.head(h)
        states=[]
        for layer,delta_head in zip(self.layers,self.delta_heads):
            h=layer(h+self.state_embed(state*g['mask']),g,state)
            delta=delta_head(h)
            state=state+delta  # eta=1 for raw normalized cx/cy and log(width/height).
            states.append(state)
        return torch.stack(states,dim=0)
