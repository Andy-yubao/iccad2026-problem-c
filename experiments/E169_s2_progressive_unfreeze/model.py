"""E168's pretrained prefix followed by two applications of a fresh S2 unit."""
import torch
from torch import nn
from .initial_model import RoleInitial
from .public import BASE, PUBLIC
from .refinement import FunctionalUnit, PublicProblemEncoder
from .shape_feedback import prepare_shape_feedback


class Model(nn.Module):
    def __init__(self, backend, width=64, ffn_dropout=.1, residual_dropout=.05):
        super().__init__()
        if not backend:
            raise ValueError('E169 requires the S0/S1 prefix and S2')
        self.initial_generator = RoleInitial(width=width, ffn_dropout=ffn_dropout,
                                             residual_dropout=residual_dropout)
        self.problem_encoder = PublicProblemEncoder(width, ffn_dropout, residual_dropout)
        self.units = nn.ModuleList([
            FunctionalUnit(width, ffn_dropout, residual_dropout, overlap_feedback=False),
            FunctionalUnit(width, ffn_dropout, residual_dropout, overlap_feedback=True),
            FunctionalUnit(width, ffn_dropout, residual_dropout, overlap_feedback=True),
        ])
        self.prefix_unfrozen = False
        self.shape_neighbor_count = None
        for module in self.prefix_modules():
            module.requires_grad_(False)
        self.train(self.training)

    def prefix_modules(self):
        return self.initial_generator, self.problem_encoder, self.units[0], self.units[1]

    def prefix_state(self):
        return {name: value for name, value in self.state_dict().items()
                if not name.startswith('units.2.')}

    def load_prefix(self, state):
        if state.keys() != self.prefix_state().keys():
            raise ValueError('Parent must contain exactly E168 L0, shared encoder, S0 and S1')
        self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.initial_generator.eval()
        if not self.prefix_unfrozen:
            self.problem_encoder.eval()
            self.units[0].eval()
            self.units[1].eval()
        return self

    def unfreeze(self):
        if self.prefix_unfrozen:
            raise RuntimeError('Prefix was already unfrozen')
        self.prefix_unfrozen = True
        for module in self.prefix_modules():
            module.requires_grad_(True)
        self.train(self.training)

    def forward(self, graph):
        if set(graph) != set(PUBLIC):
            raise ValueError('Expected registered public inputs')
        state = self.initial_generator(graph)[0]
        public = {key: graph[key] for key in BASE}
        shape_context = prepare_shape_feedback(public, self.shape_neighbor_count)
        condition = self.problem_encoder(public)
        states = [state]
        for unit in self.units:
            for _ in range(2):
                output = unit(public, state, condition, shape_context)
                states.extend(output.unbind(0))
                state = output[-1]
        return torch.stack(states)


def GeometryModel(architecture='e169', **configuration):
    if architecture != 'e169':
        raise ValueError(architecture)
    return Model(**configuration)
