"""Conditional graph critic with differentiable geometry and tied local dropout."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .batches import PUBLIC_KEYS, decode


def tied_dropout(x, probability, training, group_size):
    if not training or probability == 0:
        return x
    if x.shape[0] % group_size:
        raise ValueError('Each dropout group must contain all related draft variants')
    shape = (x.shape[0] // group_size,) + x.shape[1:]
    mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(1-probability)
    return x * (mask / (1-probability)).repeat_interleave(group_size, dim=0)


def geometry_features(state, graph):
    rectangles = decode(state, graph)
    wh = rectangles[..., 2:].clamp_min(1e-8)
    centers = rectangles[..., :2] + wh / 2
    unit = graph['area'].clamp_min(1e-12).sqrt()
    b, n, _ = state.shape
    ids = torch.arange(n, device=state.device, dtype=state.dtype)[None, :, None]
    ids = ids.expand(b, -1, -1) / graph['valid'].sum(1)[:, None, None].clamp_min(1)
    # A module's public order can affect deterministic solver tie breaks.
    node = torch.cat((graph['features'], rectangles, centers,
                      (wh / unit[..., None]).log(),
                      centers - graph['features'][..., 15:17], graph['area'][..., None], ids), -1)
    delta = centers[:, :, None] - centers[:, None, :]
    scale = ((unit[:, :, None] + unit[:, None, :]) / 2).clamp_min(1e-6)
    gap = (delta.abs() - (wh[:, :, None] + wh[:, None, :]) / 2) / scale[..., None]
    flags = graph['features'][..., 5:7]
    pair = torch.cat((delta, delta.abs(), gap, (-gap).clamp_min(0),
                      (wh[:, :, None] + wh[:, None, :]), graph['adjacency'][..., None],
                      (graph['mib'] > 0)[..., None].to(state.dtype),
                      (graph['group'] > 0)[..., None].to(state.dtype),
                      flags[:, :, None].expand(-1, -1, n, -1),
                      flags[:, None, :].expand(-1, n, -1, -1)), -1)
    assert node.shape[-1] == 29 and pair.shape[-1] == 17
    return node, pair


class CriticBlock(nn.Module):
    def __init__(self, width, heads, hidden, ffn_dropout, residual_dropout):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(5))
        self.qkv = nn.Linear(width, 3*width)
        self.attention_out = nn.Linear(width, width)
        self.edge_bias = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, heads))
        self.local = nn.ModuleList(nn.Linear(width, width) for _ in range(3))
        self.gate_up = nn.Linear(width, 2*hidden)
        self.down = nn.Linear(hidden, width)
        self.gates = nn.Parameter(torch.full((5,), math.log(.1/.9)))
        self.ffn_dropout, self.residual_dropout = ffn_dropout, residual_dropout

    def forward(self, h, pair, graph, group_size):
        b, n, width = h.shape
        q, k, v = self.qkv(self.norms[0](h)).view(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        scores = scores + self.edge_bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~graph['valid'][:, None, None, :], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, width)
        gates = self.gates.sigmoid()
        branch = tied_dropout(self.attention_out(attended), self.residual_dropout, self.training, group_size)
        h = h + gates[0] * branch
        # Each relation aggregates a pre-normalized stream; no raw h message path.
        for j, key in enumerate(('adjacency', 'mib', 'group')):
            normalized = self.norms[j+1](h)
            branch = self.local[j](graph[key] @ normalized)
            h = h + gates[j+1] * tied_dropout(branch, self.residual_dropout, self.training, group_size)
        a, value = self.gate_up(self.norms[4](h)).chunk(2, -1)
        hidden = tied_dropout(F.silu(a) * value, self.ffn_dropout, self.training, group_size)
        branch = tied_dropout(self.down(hidden), self.residual_dropout, self.training, group_size)
        return (h + gates[4] * branch) * graph['valid'][..., None]


class PostprocessorCritic(nn.Module):
    def __init__(self, width=256, layers=12, heads=8, hidden=768, pooling_queries=4,
                 ffn_dropout=.10, residual_dropout=.05, **description):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.input = nn.Sequential(nn.Linear(29, width), nn.SiLU(), nn.LayerNorm(width))
        self.blocks = nn.ModuleList(CriticBlock(width, heads, hidden, ffn_dropout, residual_dropout)
                                    for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        self.pool_queries = nn.Parameter(torch.randn(pooling_queries, width) * .02)
        self.pool_key = nn.Linear(width, width)
        self.pool_value = nn.Linear(width, width)
        self.readout = nn.Sequential(nn.Linear((pooling_queries+2)*width, 512), nn.SiLU(), nn.Linear(512, 4))
        with torch.no_grad():
            self.readout[-1].bias[0] = -3.
            self.readout[-1].bias[3] = -3.

    def forward(self, graph, state, group_size=1):
        if set(graph) - set(PUBLIC_KEYS):
            raise ValueError('Critic accepts public inputs only; labels/IDs/denominators forbidden')
        node, pair = geometry_features(state, graph)
        h = self.input(node) * graph['valid'][..., None]
        for block in self.blocks:
            h = block(h, pair, graph, group_size)
        h = self.final_norm(h)
        valid = graph['valid']
        scores = torch.einsum('qw,bnw->bqn', self.pool_queries, self.pool_key(h)) / math.sqrt(h.shape[-1])
        attention = scores.masked_fill(~valid[:, None, :], -1e4).softmax(-1)
        pooled = (attention @ self.pool_value(h)).flatten(1)
        average = (h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        maximum = h.masked_fill(~valid[..., None], -1e4).amax(1)
        raw = self.readout(torch.cat((pooled, average, maximum), -1))
        return dict(Q=1 + F.softplus(raw[:, 0]),
                    auxiliary=torch.cat((raw[:, 1:3], raw[:, 3:4].sigmoid()), -1))

    def frozen_loss(self, graph, state):
        """Future layout-training interface; caller must freeze critic parameters/eval."""
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError('Freeze the trained critic and call eval before use as a layout Loss')
        return self(graph, state)['Q']
