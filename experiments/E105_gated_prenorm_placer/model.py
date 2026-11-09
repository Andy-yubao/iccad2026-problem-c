"""Scheme1: branch-specific pre-norm, gated residuals and parameter-matched SwiGLU."""
import math
import torch
from torch import nn


class GatedGeometryBlock(nn.Module):
    def __init__(self, width, heads=4, ffn_dropout=.1, residual_dropout=.05, gate_init=.1):
        super().__init__()
        if width % heads or not 0 < gate_init < 1:
            raise ValueError('Invalid width/heads/gate initialization')
        self.heads = heads
        self.dim = width // heads
        self.use_gap = True
        self.norm_attention = nn.LayerNorm(width)
        self.norm_local = nn.LayerNorm(width)
        self.norm_ffn = nn.LayerNorm(width)
        self.norm_geometry = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.local = nn.Linear(width * 3, width)
        self.bias = nn.Sequential(nn.Linear(7,24),nn.SiLU(),nn.Linear(24,heads))
        # Same first7 initialization and neutral extra12 geometry channels as E104.
        original = self.bias[0]
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(19,24)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:,:7].copy_(original.weight)
            expanded.bias.copy_(original.bias)
        self.bias[0] = expanded
        self.ff_gate = nn.Linear(width, width * 2)
        self.ff_value = nn.Linear(width, width * 2)
        self.ff_output = nn.Linear(width * 2, width)
        self.ff_dropout = nn.Dropout(ffn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)
        # Order is attention, local, FFN, geometry feedback. Each layer owns its gates.
        self.gate_logits = nn.Parameter(torch.full((4,),math.log(gate_init/(1-gate_init))))

    def forward(self, h, g, state, geometry_feedback):
        b,n,w = h.shape
        gates = self.gate_logits.sigmoid()
        h = (h + gates[3] * self.norm_geometry(geometry_feedback)) * g['valid'][...,None]
        z = self.norm_attention(h)
        q,k,v = self.qkv(z).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
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
        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.dim) + self.bias(pair).permute(0,3,1,2)
        scores = scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended = (scores.softmax(-1) @ v).transpose(1,2).reshape(b,n,w)
        local_h = self.norm_local(h)
        local = self.local(torch.cat((g['adjacency'] @ local_h,g['mib'] @ local_h,g['group'] @ local_h),-1))
        h = h + gates[0]*self.residual_dropout(self.output(attended)) + gates[1]*self.residual_dropout(local)
        z = self.norm_ffn(h)
        gated = torch.nn.functional.silu(self.ff_gate(z)) * self.ff_value(z)
        ffn = self.ff_output(self.ff_dropout(gated))
        return (h + gates[2]*self.residual_dropout(ffn)) * g['valid'][...,None]


class GeometryModel(nn.Module):
    def __init__(self,width=128,layers=16,rectangle=True,ffn_dropout=.1,residual_dropout=.05,gate_init=.1):
        super().__init__()
        if not rectangle:
            raise ValueError('Only registered rectangle-aware scheme1 is implemented')
        self.width,self.depth = width,layers
        self.input = nn.Sequential(nn.Linear(17,width),nn.SiLU(),nn.LayerNorm(width))
        self.layers = nn.ModuleList([GatedGeometryBlock(width,ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,gate_init=gate_init) for _ in range(layers)])
        self.state_embed = nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width))
        self.head_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))

    def forward(self,g):
        if set(g)-{'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        h = self.input(g['features']) * g['valid'][...,None]
        state = self.head(self.head_norm(h))
        for layer in self.layers:
            feedback = self.state_embed(state*g['mask'])
            h = layer(h,g,state,feedback)
            state = self.head(self.head_norm(h))
        return state


def decode(state, g):
    ratio = state[..., 2].clamp(-5, 5)
    wh = torch.stack(((g['area'] * ratio.exp()).clamp_min(1e-12).sqrt(),
                      (g['area'] / ratio.exp()).clamp_min(1e-12).sqrt()), -1)
    wh = torch.where((~g['mask'][..., 2].bool())[..., None], g['specs'][..., 2:], wh)
    xy = (state[..., :2] - wh / 2).clamp_min(0)
    return torch.where((~g['mask'][..., 0].bool())[..., None], g['specs'], torch.cat((xy, wh), -1))
