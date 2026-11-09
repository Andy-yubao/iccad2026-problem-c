"""Original four geometry updates versus one delayed-readout first-draft stage."""
import math
import torch
from torch import nn
from .original_model import GeometryModel as OriginalGeometryModel

ARMS = ('original4', 'new_first')
MODEL_CONFIGS = {
    'original4': dict(width=64, blocks=4, readouts=5, final_supervision_only=True),
    'new_first': dict(width=64, blocks=4, readouts=1, swiglu_hidden=128,
                      global_conditioned_head=True, geometry_feedback=False),
}
PUBLIC = {'features', 'adjacency', 'group', 'mask', 'known', 'area',
          'specs', 'constraints', 'valid', 'scale'}


class OriginalFour(OriginalGeometryModel):
    """Unchanged local E114 arithmetic, four blocks, return only the final state."""
    def __init__(self):
        super().__init__(width=64, layers=4, rectangle=True,
                         ffn_dropout=.1, residual_dropout=.05)

    def forward(self, graph):
        return super().forward(graph)[-1]


class PublicRelationBias(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        # Nonzero norm keeps the existing relative-update protection meaningful.
        self.coefficients = nn.Parameter(torch.full((2, heads), .01))

    def forward(self, graph):
        return (self.coefficients[0][None, :, None, None]
                * graph['adjacency'][:, None]
                + self.coefficients[1][None, :, None, None]
                * (graph['group'] > 0)[:, None])


class HiddenReasoningBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.output = nn.Linear(64, 64)
        self.local = nn.Linear(128, 64)
        self.relation_bias = PublicRelationBias()
        self.ff_value = nn.Linear(64, 128)
        self.ff_gate = nn.Linear(64, 128)
        self.ff_output = nn.Linear(128, 64)
        self.ff_dropout = nn.Dropout(.1)
        self.residual_dropout = nn.Dropout(.05)

    def forward(self, h, graph):
        b, n, _ = h.shape
        u = self.norm1(h)
        q, k, v = self.qkv(u).view(b, n, 3, 4, 16).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(16) + self.relation_bias(graph)
        scores = scores.masked_fill(~graph['valid'][:, None, None, :], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, 64)
        local = self.local(torch.cat((graph['adjacency'] @ u, graph['group'] @ u), -1))
        h = h + self.residual_dropout(self.output(attended) + local)
        z = self.norm2(h)
        gated = self.ff_value(z) * torch.nn.functional.silu(self.ff_gate(z))
        h = h + self.residual_dropout(self.ff_output(self.ff_dropout(gated)))
        return h * graph['valid'][..., None]


class NewFirstDraft(nn.Module):
    """Four hidden blocks and exactly one final geometry readout."""
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.LayerNorm(64))
        self.layers = nn.ModuleList(HiddenReasoningBlock() for _ in range(4))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, graph):
        if set(graph) - PUBLIC:
            raise ValueError('Only public graph inputs may enter the network')
        valid = graph['valid'][..., None]
        h = self.input(graph['features']) * valid
        for layer in self.layers:
            h = layer(h, graph)
        context = h.sum(1) / valid.sum(1).clamp_min(1)
        context = context[:, None].expand(-1, h.shape[1], -1)
        return self.head(torch.cat((h, context), -1)) * valid


def build_model(arm):
    """Fresh common tensors; no checkpoint or cross-experiment private import."""
    if arm not in ARMS:
        raise ValueError(arm)
    original = OriginalFour()
    if arm == 'original4':
        return original
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261108116)
        candidate = NewFirstDraft()
    candidate.input.load_state_dict(original.input.state_dict())
    for old, new in zip(original.layers, candidate.layers):
        for name in ('norm1', 'norm2', 'qkv', 'output', 'local'):
            getattr(new, name).load_state_dict(getattr(old, name).state_dict())
    return candidate
