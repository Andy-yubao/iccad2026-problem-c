"""One relation-row encoding, four hidden blocks, exactly one geometry readout."""
import math
import torch
from torch import nn

ARMS = ('full', 'pure')
PUBLIC = {'features', 'adjacency', 'group', 'constraints', 'valid', 'address',
          'mask', 'area', 'known', 'specs', 'scale'}


def relation_descriptor(g):
    """Destination columns and self addresses use the same bijection.

    Validation of integer addresses/memberships is done on CPU before capture.
    No labels, predicted geometry, or hidden-neighbor features enter here.
    """
    if set(g) - PUBLIC:
        raise ValueError('Only public problem fields and temporary addresses are allowed')
    f, valid, address = g['features'], g['valid'], g['address']
    b, n, channels = f.shape
    if channels != 16 or n > 128 or address.shape != (b, n):
        raise ValueError('Unexpected relation-row dimensions')
    active = valid[:, :, None] & valid[:, None, :]
    identity = (address[..., None] == torch.arange(128, device=f.device)).to(f.dtype)
    identity = identity * valid[..., None]
    index = address[:, None, :].expand(-1, n, -1)
    wire = torch.zeros((b, n, 128), device=f.device, dtype=f.dtype).scatter(
        2, index, g['adjacency'] * active)
    labels = g['constraints'][..., 3]
    same = (labels[:, :, None] == labels[:, None, :]) & (labels[:, :, None] > 0) & active
    grouping = torch.zeros_like(wire).scatter(2, index, same.to(f.dtype))
    return torch.cat((f, identity, wire, grouping), -1) * valid[..., None]


class PureBlock(nn.Module):
    """No relation matrices can be passed to this block."""
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.output = nn.Linear(64, 64)
        self.ff_value = nn.Linear(64, 128)
        self.ff_gate = nn.Linear(64, 128)
        self.ff_output = nn.Linear(128, 64)
        self.ff_dropout = nn.Dropout(.1)
        self.residual_dropout = nn.Dropout(.05)

    def attention(self, h, valid, bias=None):
        b, n, _ = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, 4, 16).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(16)
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        return (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, 64)

    def finish(self, h, valid):
        z = self.norm2(h)
        hidden = self.ff_value(z) * torch.nn.functional.silu(self.ff_gate(z))
        return (h + self.residual_dropout(self.ff_output(self.ff_dropout(hidden)))) * valid[..., None]

    def forward(self, h, valid):
        h = h + self.residual_dropout(self.output(self.attention(h, valid)))
        return self.finish(h, valid)


class PublicRelationBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.full((2, 4), .01))

    def forward(self, adjacency, group):
        return (self.coefficients[0][None, :, None, None] * adjacency[:, None]
                + self.coefficients[1][None, :, None, None] * (group > 0)[:, None])


class FullBlock(PureBlock):
    def __init__(self, index):
        super().__init__()
        # Extra branches do not change common-core initialization or CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261112131 + index)
            self.local = nn.Linear(128, 64)
            self.relation_bias = PublicRelationBias()

    def forward(self, h, valid, adjacency, group):
        u = self.norm1(h)
        local = self.local(torch.cat((adjacency @ u, group @ u), -1))
        attended = self.attention(h, valid, self.relation_bias(adjacency, group))
        h = h + self.residual_dropout(self.output(attended) + local)
        return self.finish(h, valid)


class GeometryModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.input = nn.Sequential(nn.Linear(400, 128), nn.SiLU(), nn.Linear(128, 64), nn.LayerNorm(64))
        self.layers = nn.ModuleList(FullBlock(i) if arm == 'full' else PureBlock() for i in range(4))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, g):
        valid = g['valid']
        h = self.input(relation_descriptor(g)) * valid[..., None]
        for layer in self.layers:
            if self.arm == 'full':
                h = layer(h, valid, g['adjacency'], g['group'])
            else:
                h = layer(h, valid)
        context = h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        context = context[:, None].expand(-1, h.shape[1], -1)
        return self.head(torch.cat((h, context), -1)) * valid[..., None]


def build_model(arm, seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return GeometryModel(arm)
