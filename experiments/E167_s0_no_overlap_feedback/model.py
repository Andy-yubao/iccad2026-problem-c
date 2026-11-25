"""Frozen E165 L0 and two applications of E166's shared S0 unit."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .public import BASE, PUBLIC
from .refinement import FunctionalUnit, PublicProblemEncoder


class Model(nn.Module):
    def __init__(self, backend, width=64, ffn_dropout=.1, residual_dropout=.05):
        super().__init__()
        if not backend:
            raise ValueError('E167 trains S0 with a frozen L0')
        self.initial_generator = RoleInitial(width=width, ffn_dropout=ffn_dropout,
                                             residual_dropout=residual_dropout)
        self.initial_generator.requires_grad_(False).eval()
        self.problem_encoder = PublicProblemEncoder(width, ffn_dropout, residual_dropout)
        self.units = nn.ModuleList([
            FunctionalUnit(width, ffn_dropout, residual_dropout, overlap_feedback=False)
        ])

    def train(self, mode=True):
        super().train(mode)
        self.initial_generator.eval()
        return self

    def forward(self, graph):
        if set(graph) != set(PUBLIC):
            raise ValueError('Expected registered public inputs')
        with torch.no_grad():
            state = self.initial_generator(graph)[0]
        public = {key: graph[key] for key in BASE}
        condition = self.problem_encoder(public)
        states = [state]
        for _ in range(2):
            output = self.units[0](public, state, condition)
            states.extend(output.unbind(0))
            state = output[-1]
        return torch.stack(states)


def GeometryModel(architecture='e167', **configuration):
    if architecture != 'e167':
        raise ValueError(architecture)
    return Model(**configuration)
