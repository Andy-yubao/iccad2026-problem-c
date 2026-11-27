"""Four-layer attention with public MIB-consistent shapes during training/inference."""
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
    def __init__(self, width=96, layers=4):
        super().__init__()
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
        state = consistent_ratio(self.head(h),g)
        for layer in self.layers:
            h = layer(h + self.state_embed(state * g['mask']), g, state)
            state = consistent_ratio(self.head(h),g)
        return state


def mib_classes(g):
    groups=g['constraints'][...,2]
    return (groups[:,:,None]==groups[:,None,:])&(groups[:,:,None]>0)&g['valid'][:,:,None]&g['valid'][:,None,:]


def consistent_ratio(state,g):
    same=mib_classes(g);weights=same.to(state.dtype)
    fixed=(~g['mask'][...,2].bool())&g['valid'];fixed_members=same&fixed[:,None,:]
    values=torch.where(fixed,g['known'][...,2],state[...,2])
    pooled=(weights@values[...,None]).squeeze(-1)/weights.sum(-1).clamp_min(1)
    first=fixed_members.to(state.dtype).argmax(-1)
    public=torch.gather(g['known'][...,2],1,first)
    group_ratio=torch.where(fixed_members.any(-1),public,pooled)
    ratio=torch.where(same.any(-1),group_ratio,values)
    return torch.cat((state[...,:2],ratio[...,None]),-1)


def decode(state, g):
    ratio = state[..., 2].clamp(-5, 5)
    wh = torch.stack(((g['area'] * ratio.exp()).clamp_min(1e-12).sqrt(),
                      (g['area'] / ratio.exp()).clamp_min(1e-12).sqrt()), -1)
    wh = torch.where((~g['mask'][..., 2].bool())[..., None], g['specs'][..., 2:], wh)
    same=mib_classes(g);fixed=(~g['mask'][...,2].bool())&g['valid'];members=same&fixed[:,None,:]
    first=members.to(state.dtype).argmax(-1)
    public_wh=torch.gather(g['specs'][...,2:],1,first[...,None].expand(-1,-1,2))
    wh=torch.where(members.any(-1)[...,None],public_wh,wh)
    xy = (state[..., :2] - wh / 2).clamp_min(0)
    return torch.where((~g['mask'][..., 0].bool())[..., None], g['specs'], torch.cat((xy, wh), -1))


def physical_mib(positions,inputs):
    """Restore exact public WH after normalized float32 decode, preserving centers."""
    import numpy as np
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    for group in np.unique(c[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(c[:,2]==group);rigid=ids[locked[ids]];free=ids[~locked[ids]]
        if len(rigid) and len(free):
            centers=positions[free,:2]+positions[free,2:]/2
            positions[free,2:]=s[rigid[0],2:]
            positions[free,:2]=np.maximum(0,centers-positions[free,2:]/2)
    return positions
