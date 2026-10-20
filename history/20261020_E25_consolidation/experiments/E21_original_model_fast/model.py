"""Existing 4-layer geometry attention, random initialization only."""
import math
import torch
from torch import nn


class GeometryBlock(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.local = nn.Linear(width * 3, width)
        self.bias = nn.Sequential(nn.Linear(7, 24), nn.SiLU(), nn.Linear(24, heads))
        self.ff = nn.Sequential(nn.Linear(width, width * 3), nn.SiLU(), nn.Linear(width * 3, width))

    def forward(self, h, g, state):
        b, n, w = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        xy = torch.where(g['mask'][..., :2].bool(), state[..., :2], g['known'][..., :2])
        delta = xy[:, :, None] - xy[:, None, :]
        pair = torch.cat((delta, delta.abs(), g['adjacency'][..., None],
                          (g['mib'] > 0)[..., None], (g['group'] > 0)[..., None]), -1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim) + self.bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~g['valid'][:, None, None, :], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, w)
        local = self.local(torch.cat((g['adjacency'] @ h, g['mib'] @ h, g['group'] @ h), -1))
        h = h + self.output(attended) + local
        return (h + self.ff(self.norm2(h))) * g['valid'][..., None]


class GeometryModel(nn.Module):
    def __init__(self, width=96, layers=4,architecture='attention'):
        super().__init__()
        if architecture!='attention':raise ValueError('Only the retained attention baseline is supported')
        self.width, self.depth = width, layers
        self.input = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList([GeometryBlock(width) for _ in range(layers)])
        self.state_embed = nn.Linear(3, width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, g):
        if set(g) - {'features', 'adjacency', 'mib', 'group', 'mask', 'known', 'area',
                     'specs', 'constraints', 'valid', 'scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        h = self.input(g['features']) * g['valid'][..., None]
        state = self.head(h)
        for layer in self.layers:
            h = layer(h + self.state_embed(state * g['mask']), g, state)
            state = self.head(h)
        return state


def decode(state, g):
    ratio = state[..., 2].clamp(-5, 5)
    wh = torch.stack(((g['area'] * ratio.exp()).clamp_min(1e-12).sqrt(),
                      (g['area'] / ratio.exp()).clamp_min(1e-12).sqrt()), -1)
    wh = torch.where((~g['mask'][..., 2].bool())[..., None], g['specs'][..., 2:], wh)
    xy = (state[..., :2] - wh / 2).clamp_min(0)
    return torch.where((~g['mask'][..., 0].bool())[..., None], g['specs'], torch.cat((xy, wh), -1))
