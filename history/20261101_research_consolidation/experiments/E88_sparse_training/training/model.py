"""E39 rectangle model with dropout only inside each FFN hidden activation."""
import math
import torch
from torch import nn


class GeometryBlock(nn.Module):
    def __init__(self, width, heads=4, ffn_dropout=0.):
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
        # Separate parameter-free module preserves the original ff.0 / ff.2 keys
        # and consumes no initialization RNG. Public inputs and decoded specs stay intact.
        self.ff_dropout = nn.Dropout(ffn_dropout)

    def feed_forward(self, h):
        return self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](h))))

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
        return (h + self.feed_forward(self.norm2(h))) * g['valid'][..., None]


class GeometryModel(nn.Module):
    def __init__(self, width=96, layers=4, rectangle=False, use_gap=True, ffn_dropout=0.):
        super().__init__()
        self.width, self.depth = width, layers
        self.input = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList([RectangleBlock(width, use_gap=use_gap, ffn_dropout=ffn_dropout) if rectangle else GeometryBlock(width, ffn_dropout=ffn_dropout)
                                     for _ in range(layers)])
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


class RectangleBlock(GeometryBlock):
    """Same trunk; enrich the small pair-bias network with decoded rectangles."""
    def __init__(self, width, heads=4, use_gap=True, ffn_dropout=0.):
        super().__init__(width, heads, ffn_dropout=ffn_dropout)
        self.use_gap = use_gap
        # Preserve the ordinary trunk's initialization and RNG stream. The new
        # public geometry channels start neutral and acquire weights by learning.
        original = self.bias[0]
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(19, 24)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :7].copy_(original.weight)
            expanded.bias.copy_(original.bias)
        self.bias[0] = expanded

    def forward(self, h, g, state):
        b, n, w = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        rectangles = decode(state, g)
        wh = rectangles[..., 2:]
        xy = rectangles[..., :2]+wh/2
        delta = xy[:, :, None]-xy[:, None, :]
        original_xy = torch.where(g['mask'][..., :2].bool(), state[..., :2], g['known'][..., :2])
        original_delta = original_xy[:, :, None]-original_xy[:, None, :]
        pair = torch.cat((original_delta, original_delta.abs(), g['adjacency'][..., None],
                          (g['mib'] > 0)[..., None], (g['group'] > 0)[..., None]), -1)
        local = g['area'].clamp_min(1e-12).sqrt()
        pair_scale = ((local[:, :, None]+local[:, None, :])/2).clamp_min(1e-6)
        normalized_wh = wh/local[..., None]
        gap = (delta.abs()-(wh[:, :, None]+wh[:, None, :])/2)/pair_scale[..., None]
        penetration = (-gap).clamp_min(0)
        if not self.use_gap:
            gap = gap*0; penetration = penetration*0
        flags = g['constraints'][..., :2].to(h.dtype)
        dimensions = torch.cat((normalized_wh[:, :, None].expand(-1, -1, n, -1),
                                normalized_wh[:, None, :].expand(-1, n, -1, -1)), -1)
        anchors = torch.cat((flags[:, :, None].expand(-1, -1, n, -1),
                             flags[:, None, :].expand(-1, n, -1, -1)), -1)
        pair = torch.cat((pair, dimensions, gap, penetration, anchors), -1)
        scores = (q @ k.transpose(-2, -1))/math.sqrt(self.dim)+self.bias(pair).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~g['valid'][:, None, None, :], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, w)
        aggregated = self.local(torch.cat((g['adjacency'] @ h, g['mib'] @ h, g['group'] @ h), -1))
        h = h+self.output(attended)+aggregated
        return (h+self.feed_forward(self.norm2(h)))*g['valid'][..., None]


def decode(state, g):
    ratio = state[..., 2].clamp(-5, 5)
    wh = torch.stack(((g['area'] * ratio.exp()).clamp_min(1e-12).sqrt(),
                      (g['area'] / ratio.exp()).clamp_min(1e-12).sqrt()), -1)
    wh = torch.where((~g['mask'][..., 2].bool())[..., None], g['specs'][..., 2:], wh)
    xy = (state[..., :2] - wh / 2).clamp_min(0)
    return torch.where((~g['mask'][..., 0].bool())[..., None], g['specs'], torch.cat((xy, wh), -1))
