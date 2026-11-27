"""Public-only pair message values, independently maintained from the E42 trunk."""
import torch
from torch import nn
from .base_model import GeometryModel as BaseModel, decode


class PairMessage(nn.Module):
    def __init__(self, width, hidden, mode, scale):
        super().__init__()
        self.mode, self.scale = mode, scale
        self.norm = nn.LayerNorm(width)
        self.receiver = nn.Linear(width, hidden)
        self.sender = nn.Linear(width, hidden, bias=False)
        self.edge = nn.Linear(19, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.output = nn.Linear(hidden, width)
        # New path starts neutral; old trunk parameters and its initialization RNG stay intact.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def edge_features(self, state, g):
        p = decode(state, g)
        wh = p[..., 2:]
        center = p[..., :2]+wh/2
        local = g['area'].clamp_min(1e-12).sqrt()
        size = ((local[:, :, None]+local[:, None, :])/2).clamp_min(1e-6)
        delta = (center[:, :, None]-center[:, None, :])/size[..., None]
        gap = delta.abs()-(wh[:, :, None]+wh[:, None, :])/(2*size[..., None])
        shape = (wh/local[..., None].clamp_min(1e-6)).clamp_min(1e-6).log().clamp(-5, 5)
        n = state.shape[1]
        geometry = torch.cat((delta.clamp(-8, 8), delta.abs().clamp_max(8), gap.clamp(-8, 8),
                              (-gap).clamp(0, 8), shape[:, :, None].expand(-1, -1, n, -1),
                              shape[:, None, :].expand(-1, n, -1, -1)), -1)
        if self.mode == 'features':
            # Same parameter tensors and execution shape; zero explicit geometric edge channels.
            geometry = geometry*0
        flags = g['constraints'][..., :2].to(state.dtype)
        relations = torch.stack((g['adjacency'], (g['mib'] > 0).to(state.dtype),
                                 (g['group'] > 0).to(state.dtype)), -1)
        return torch.cat((geometry, flags[:, :, None].expand(-1, -1, n, -1),
                          flags[:, None, :].expand(-1, n, -1, -1), relations), -1)

    def forward(self, h, g, state):
        normalized = self.norm(h)
        value = torch.nn.functional.silu(self.receiver(normalized)[:, :, None]+
                                         self.sender(normalized)[:, None, :]+self.edge(self.edge_features(state, g)))
        n = state.shape[1]
        valid = g['valid'][:, :, None] & g['valid'][:, None, :]
        valid = valid & ~torch.eye(n, dtype=torch.bool, device=state.device)[None]
        score = self.score(value).squeeze(-1).masked_fill(~valid, -1e4)
        attention = score.softmax(-1)*valid
        attention = attention/attention.sum(-1, keepdim=True).clamp_min(1e-8)
        aggregated = (value*attention[..., None]).sum(2)
        return self.scale*self.output(aggregated)*g['valid'][..., None]


class GeometryModel(BaseModel):
    def __init__(self, width=96, layers=12, rectangle=True, ffn_dropout=.05,
                 message_mode='geometry', message_width=24, message_scale=.1):
        super().__init__(width=width, layers=layers, rectangle=rectangle, ffn_dropout=ffn_dropout)
        if message_mode not in ('geometry', 'features', 'none'):
            raise ValueError(message_mode)
        self.message_mode = message_mode
        # Preserve the base model's post-initialization RNG, including paired Dropout draws.
        with torch.random.fork_rng(devices=[]):
            self.messages = nn.ModuleList([] if message_mode == 'none' else
                [PairMessage(width, message_width, message_mode, message_scale) for _ in range(layers)])

    def forward(self, g):
        allowed = {'features', 'adjacency', 'mib', 'group', 'mask', 'known', 'area',
                   'specs', 'constraints', 'valid', 'scale'}
        if set(g)-allowed:
            raise ValueError('Only public graph inputs may enter the network')
        h = self.input(g['features'])*g['valid'][..., None]
        state = self.head(h)
        for index, layer in enumerate(self.layers):
            incoming = h+self.state_embed(state*g['mask'])
            h = layer(incoming, g, state)
            if self.message_mode != 'none':
                h = (h+self.messages[index](incoming, g, state))*g['valid'][..., None]
            state = self.head(h)
        return state
