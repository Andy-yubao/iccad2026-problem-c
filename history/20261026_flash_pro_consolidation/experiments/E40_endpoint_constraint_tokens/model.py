"""Persistent constraint tokens and loss-preserving public endpoint interactions."""
import math
import torch
from torch import nn
from .reference_model import GeometryModel as PlainModel, decode
from .public import EXTRA_KEYS


class TokenInteraction(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pin_encoder = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.pin_norm = nn.LayerNorm(width); self.pin_output = nn.Linear(width, width)
        self.types = nn.Embedding(2, width)
        self.node_norm = nn.LayerNorm(width); self.token_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width); self.key = nn.Linear(width, width)
        self.token_update = nn.Linear(width, width); self.geometry = nn.Linear(4, width)
        self.token_ff = nn.Sequential(nn.Linear(width, width*2), nn.SiLU(), nn.Linear(width*2, width))
        self.token_output = nn.Linear(width, width)
        self.pin_gate = nn.Parameter(torch.zeros(())); self.constraint_gate = nn.Parameter(torch.zeros(()))

    def forward(self, h, graph, state, tokens, pooled):
        rectangles = decode(state, graph); center = rectangles[..., :2]+rectangles[..., 2:]/2
        local = graph['area'].clamp_min(1e-12).sqrt()
        weights = graph['terminal_weight']*graph['terminal_valid'][:, None]
        if pooled:
            centroid = weights @ graph['terminal_xy']
            delta = (centroid-center)/local[..., None]
            endpoint = self.pin_encoder(torch.cat((delta, delta.abs()), -1))
        else:
            bi, ni, pi = graph['terminal_edge_index'].unbind(1)
            delta = (graph['terminal_xy'][bi, pi]-center[bi, ni])/local[bi, ni, None]
            encoded = self.pin_encoder(torch.cat((delta, delta.abs()), -1))
            b, n, width = h.shape
            endpoint = h.new_zeros((b*n, width)).index_add(0, bi*n+ni,
                encoded*graph['terminal_edge_weight'][:, None]).reshape(b, n, width)
        endpoint = self.pin_output(self.pin_norm(endpoint))*(weights.sum(-1) > 0)[..., None]
        member = graph['constraint_member']*graph['valid'][:, None]
        normalized = member/member.sum(-1, keepdim=True).clamp_min(1)
        valid = graph['constraint_valid'][..., None]
        if tokens is None:
            tokens = (normalized @ h+self.types(graph['constraint_type']))*valid
        positions = torch.cat((center, rectangles[..., 2:]), -1)
        query = self.query(self.token_norm(tokens)); key = self.key(self.node_norm(h))
        scores = (query @ key.transpose(-2, -1))/math.sqrt(h.shape[-1])
        attention = scores.masked_fill(member <= 0, -1e4).softmax(-1)*member
        attention = attention/attention.sum(-1, keepdim=True).clamp_min(1e-8)
        tokens = (tokens+self.token_update(attention @ h)+self.geometry(normalized @ positions))*valid
        tokens = (tokens+self.token_ff(self.token_norm(tokens)))*valid
        reverse = member.transpose(1, 2)
        reverse = reverse/reverse.sum(-1, keepdim=True).clamp_min(1)
        constraints = reverse @ self.token_output(self.token_norm(tokens))
        output = h+self.pin_gate.tanh()*endpoint+self.constraint_gate.tanh()*constraints
        return output*graph['valid'][..., None], tokens


class GeometryModel(PlainModel):
    def __init__(self, width=96, layers=4, mode='plain'):
        super().__init__(width, layers)
        if mode not in ('plain', 'full', 'pooled'):
            raise ValueError(mode)
        self.mode = mode
        if mode != 'plain':
            with torch.random.fork_rng(devices=[]):
                self.interactions = nn.ModuleList([TokenInteraction(width) for _ in range(layers)])

    def forward(self, graph):
        plain = {key: value for key, value in graph.items() if key not in EXTRA_KEYS}
        if self.mode == 'plain':
            return super().forward(plain)
        allowed = {'features', 'adjacency', 'mib', 'group', 'mask', 'known', 'area',
                   'specs', 'constraints', 'valid', 'scale'}
        if set(plain)-allowed or any(key not in graph for key in EXTRA_KEYS):
            raise ValueError('Only complete public graph inputs may enter token network')
        h = self.input(graph['features'])*graph['valid'][..., None]
        state = self.head(h); tokens = None
        for layer, interaction in zip(self.layers, self.interactions):
            h = layer(h+self.state_embed(state*graph['mask']), graph, state)
            intermediate = self.head(h)
            h, tokens = interaction(h, graph, intermediate, tokens, self.mode == 'pooled')
            state = self.head(h)
        return state
