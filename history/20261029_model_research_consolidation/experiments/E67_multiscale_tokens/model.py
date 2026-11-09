"""Original module trunk plus persistent grouping/global and MIB-shape tokens."""
import math
import torch
from torch import nn
from .base_model import GeometryModel as BaseModel, decode


def incidence(labels, valid, mode):
    """One token per positive public group; slots are representatives, never ID embeddings."""
    n = labels.shape[1]
    eligible = (labels > 0) & valid
    same = (labels[:, :, None] == labels[:, None, :]) & eligible[:, :, None] & eligible[:, None, :]
    index = torch.arange(n, device=labels.device)
    representative = torch.where(same, index[None, None, :], n).amin(-1)
    assignment = ((representative[:, :, None] == index[None, None, :]) & eligible[:, :, None]).float()
    active = assignment.sum(1) > 0
    if mode == 'global':
        # Same eligible modules and active-token count, but no true membership in the new branch.
        assignment = eligible[..., None].float()*active[:, None, :]/active.sum(1).clamp_min(1)[:, None, None]
    return assignment, active


def pool(assignment, values):
    return (assignment.transpose(1, 2) @ values)/assignment.sum(1).clamp_min(1)[..., None]


class ScaleExchange(nn.Module):
    def __init__(self, width, token_width, scale):
        super().__init__()
        self.width, self.scale = token_width, scale
        self.node_norm = nn.LayerNorm(width)
        self.project = nn.Linear(width, token_width)
        self.types = nn.Parameter(torch.zeros(2, token_width))
        self.token_norm = nn.LayerNorm(token_width)
        self.qkv = nn.Linear(token_width, 3*token_width)
        self.attended = nn.Linear(token_width, token_width)
        self.ff_norm = nn.LayerNorm(token_width)
        self.ff = nn.Sequential(nn.Linear(token_width, 2*token_width), nn.SiLU(), nn.Linear(2*token_width, token_width))
        self.group_output = nn.Linear(2*token_width, width, bias=False)
        self.shape_project = nn.Linear(2, token_width)
        self.shape_norm = nn.LayerNorm(token_width)
        self.shape_update = nn.Sequential(nn.Linear(token_width, token_width), nn.SiLU(), nn.Linear(token_width, token_width))
        self.shape_output = nn.Linear(token_width, width, bias=False)
        nn.init.zeros_(self.group_output.weight)
        nn.init.zeros_(self.shape_output.weight)

    def shape_inputs(self, state, g):
        # Fixed/preplaced shapes come only from their public specification.
        ratio = torch.where(g['mask'][..., 2].bool(), state[..., 2], g['known'][..., 2])
        return torch.stack((ratio.clamp(-5, 5), g['area'].clamp_min(1e-12).log().clamp(-20, 5)), -1)

    def forward(self, h, g, state, assignments, context):
        group, group_valid, mib, mib_valid = assignments
        global_previous, group_previous, shape_previous = context
        projected = self.project(self.node_norm(h))*g['valid'][..., None]
        global_pool = projected.sum(1, keepdim=True)/g['valid'].sum(1).clamp_min(1)[:, None, None]
        group_pool = pool(group, projected)
        tokens = torch.cat((global_previous+global_pool+self.types[0],
                            group_previous+group_pool+self.types[1]), 1)
        valid = torch.cat((torch.ones_like(group_valid[:, :1]), group_valid), 1)
        b, length, width = tokens.shape
        heads = 4
        q, k, v = self.qkv(self.token_norm(tokens)).view(b, length, 3, heads, width//heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1))/math.sqrt(width//heads)
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, length, width)
        tokens = tokens+self.attended(attended)
        tokens = (tokens+self.ff(self.ff_norm(tokens)))*valid[..., None]
        global_next, group_next = tokens[:, :1], tokens[:, 1:]
        group_to_nodes = group @ group_next
        coarse = self.group_output(torch.cat((global_next.expand(-1, h.shape[1], -1), group_to_nodes), -1))
        # MIB state has a separate shape-only update, with no pooled centers or group/global tokens.
        shape_pool = pool(mib, self.shape_project(self.shape_inputs(state, g)))
        shape_next = (shape_previous+self.shape_update(self.shape_norm(shape_previous+shape_pool)))*mib_valid[..., None]
        shape_to_nodes = self.shape_output(mib @ shape_next)
        updated = (h+self.scale*(coarse+shape_to_nodes))*g['valid'][..., None]
        return updated, (global_next, group_next, shape_next)


class GeometryModel(BaseModel):
    def __init__(self, width=96, layers=12, rectangle=True, ffn_dropout=.05,
                 pool_mode='groups', token_width=24, token_scale=.1):
        super().__init__(width=width, layers=layers, rectangle=rectangle, ffn_dropout=ffn_dropout)
        if pool_mode not in ('groups', 'global') or token_width % 4:
            raise ValueError('E67 pool_mode/token width mismatch')
        self.pool_mode, self.token_width = pool_mode, token_width
        with torch.random.fork_rng(devices=[]):
            self.scales = nn.ModuleList([ScaleExchange(width, token_width, token_scale) for _ in range(layers)])

    def assignments(self, g):
        group, group_valid = incidence(g['constraints'][..., 3], g['valid'], self.pool_mode)
        mib, mib_valid = incidence(g['constraints'][..., 2], g['valid'], self.pool_mode)
        return group, group_valid, mib, mib_valid

    def forward(self, g):
        allowed = {'features', 'adjacency', 'mib', 'group', 'mask', 'known', 'area',
                   'specs', 'constraints', 'valid', 'scale'}
        if set(g)-allowed:
            raise ValueError('Only public graph inputs may enter the network')
        h = self.input(g['features'])*g['valid'][..., None]
        state = self.head(h)
        assignments = self.assignments(g)
        b, n, _ = h.shape
        context = (h.new_zeros(b, 1, self.token_width), h.new_zeros(b, n, self.token_width),
                   h.new_zeros(b, n, self.token_width))
        for layer, exchange in zip(self.layers, self.scales):
            h = layer(h+self.state_embed(state*g['mask']), g, state)
            h, context = exchange(h, g, state, assignments, context)
            state = self.head(h)
        return state
