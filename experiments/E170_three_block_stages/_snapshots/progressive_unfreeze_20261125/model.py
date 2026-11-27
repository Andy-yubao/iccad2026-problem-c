"""Frozen L0 initializer followed by independent three-block refinement stages."""
import math

import torch
from torch import nn

from .geometry_feedback import FEATURE_NAMES, module_features, pair_features
from .initial_model import RoleInitial
from .public import BASE, PUBLIC
from .refinement import CoordinatedBlock, ResidualHead
from .shape_feedback import FEATURE_NAMES as SHAPE_NAMES
from .shape_feedback import prepare_shape_feedback, shape_features


class AttentionBlock(nn.Module):
    def __init__(self, width, ffn_dropout, residual_dropout):
        super().__init__()
        self.heads = 4
        self.head_width = width // self.heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3 * width), nn.SiLU(),
                                 nn.Dropout(ffn_dropout), nn.Linear(3 * width, width))
        self.residual = nn.Dropout(residual_dropout)

    def forward(self, hidden, valid):
        batch, nodes, width = hidden.shape
        q, k, v = self.qkv(self.norm(hidden)).view(
            batch, nodes, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        attention = scores.masked_fill(~valid[:, None, None, :], -1e4).softmax(-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, nodes, width)
        hidden = hidden + self.residual(self.output(attended))
        return (hidden + self.residual(self.ffn(hidden))) * valid[..., None]


class RefinementStage(nn.Module):
    def __init__(self, width, ffn_dropout, residual_dropout, overlap_feedback):
        super().__init__()
        self.overlap_feedback = overlap_feedback
        self.public_embed = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.LayerNorm(width))
        self.layout_embed = nn.Linear(3, width)
        demand_width = 16 + len(FEATURE_NAMES) + len(SHAPE_NAMES)
        self.demand = nn.Sequential(nn.Linear(demand_width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.LayerNorm(width))
        self.geometry_attention = CoordinatedBlock(width, ffn_dropout, residual_dropout)
        self.attention = nn.ModuleList(AttentionBlock(width, ffn_dropout, residual_dropout) for _ in range(2))
        self.head = ResidualHead(width)

    def forward(self, graph, layout, shape_context):
        valid = graph['valid']
        pair, boxes = pair_features(layout, graph, self.overlap_feedback)
        dynamic = module_features(boxes, graph, self.overlap_feedback)
        response = shape_features(boxes, shape_context, self.overlap_feedback)
        demand = self.demand(torch.cat((dynamic, response, graph['features']), dim=-1)) * valid[..., None]
        hidden = (self.public_embed(graph['features']) + self.layout_embed(layout)) * valid[..., None]
        hidden = self.geometry_attention(hidden, graph, demand, pair)
        for block in self.attention:
            hidden = block(hidden, valid)
        return self.head(hidden, layout, graph)


class GeometryModel(nn.Module):
    def __init__(self, stage, width=64, ffn_dropout=.1, residual_dropout=.05,
                 overlap_feedback=True):
        super().__init__()
        if stage not in range(1, 5):
            raise ValueError('Expected E170 stage 1–4')
        self.stage = stage
        self.initial_generator = RoleInitial(width=width, ffn_dropout=ffn_dropout,
                                             residual_dropout=residual_dropout)
        self.stages = nn.ModuleList(RefinementStage(width, ffn_dropout, residual_dropout, overlap_feedback)
                                    for _ in range(stage))
        self.prefix_unfrozen = False
        self.shape_neighbor_count = None
        for module in self.prefix_modules():
            module.requires_grad_(False)
        self.train(self.training)

    def prefix_modules(self):
        return (self.initial_generator, *list(self.stages)[:-1])

    def prefix_state(self):
        newest = f'stages.{self.stage - 1}.'
        return {name: value for name, value in self.state_dict().items() if not name.startswith(newest)}

    def load_prefix(self, state):
        if state.keys() != self.prefix_state().keys():
            raise ValueError(f'Parent weights do not match the S0–S{self.stage - 1} prefix')
        self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if not self.prefix_unfrozen:
            for module in self.prefix_modules():
                module.eval()
        return self

    def unfreeze(self):
        if self.prefix_unfrozen:
            raise RuntimeError('Prefix is already unfrozen')
        self.prefix_unfrozen = True
        for module in self.prefix_modules():
            module.requires_grad_(True)
        self.train(self.training)

    def forward(self, graph):
        if set(graph) != set(PUBLIC):
            raise ValueError('Expected registered public inputs')
        public = {key: graph[key] for key in BASE}
        context = prepare_shape_feedback(public, self.shape_neighbor_count)
        state = self.initial_generator(graph)[0]
        states = [state]
        for stage in self.stages:
            state = stage(public, state, context)
            states.append(state)
        return torch.stack(states)
