"""Three-block L0 and E151's three shared two-layer refinement units."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .public import BASE, PUBLIC
from .refinement import FunctionalUnit, PublicProblemEncoder


class Model(nn.Module):
    unit_order = (0, 0, 1, 1, 2, 2)

    def __init__(self, backend, width=64, ffn_dropout=.1, residual_dropout=.05):
        super().__init__()
        self.backend = backend
        self.initial_generator = RoleInitial(width=width, ffn_dropout=ffn_dropout,
                                             residual_dropout=residual_dropout)
        if backend:
            self.initial_generator.requires_grad_(False).eval()
            self.problem_encoder = PublicProblemEncoder(width, ffn_dropout, residual_dropout)
            self.units = nn.ModuleList(FunctionalUnit(width, ffn_dropout, residual_dropout)
                                       for _ in range(3))

    def train(self, mode=True):
        super().train(mode)
        if self.backend:
            self.initial_generator.eval()
        return self

    def unfreeze(self):
        self.initial_generator.requires_grad_(True)

    def forward(self, graph):
        if set(graph) != set(PUBLIC):
            raise ValueError('Expected registered public inputs')
        if not self.backend:
            return self.initial_generator(graph)
        state = self.initial_generator(graph)[0]
        public = {key: graph[key] for key in BASE}
        condition = self.problem_encoder(public)
        states = [state]
        for index in self.unit_order:
            output = self.units[index](public, state, condition)
            states.extend(output.unbind(0))
            state = output[-1]
        return torch.stack(states)


def GeometryModel(architecture='e165', **configuration):
    if architecture != 'e165':
        raise ValueError(architecture)
    return Model(**configuration)
