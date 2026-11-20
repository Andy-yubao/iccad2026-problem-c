"""Node26 geometry in Q/K; all other E114 trunk tensors start identically."""
import math
import torch
from torch import nn
from .original_model import GeometryModel as OriginalModel, GeometryBlock, decode

FEATURE_NAMES = ('cx','cy','cx2','cy2','left','right','bottom','top','width','height',
                 'width_local','height_local','log_ratio','sqrt_area','raw_cx','raw_cy',
                 'wire_cx','wire_cy','wire_w','wire_h','group_cx','group_cy','group_w','group_h',
                 'has_wire','has_group')
AXIS_SWAP = (1,0,3,2,6,7,4,5,9,8,11,10,12,13,15,14,17,16,19,18,21,20,23,22,24,25)

def node_geometry(state, g):
    """26 O(N) geometry channels in original normalized coordinates, each layer."""
    rect = decode(state, g)
    wh = rect[..., 2:]
    center = rect[..., :2] + wh / 2
    local = g['area'].clamp_min(1e-12).sqrt()
    edges = torch.stack((rect[...,0], rect[...,0]+wh[...,0],
                         rect[...,1], rect[...,1]+wh[...,1]), -1)
    own = torch.cat((center, center.square(), edges, wh, wh/local[...,None],
                     (wh[...,0]/wh[...,1]).log()[...,None], local[...,None], state[...,:2]), -1)
    geometry = torch.cat((center, wh), -1) * g['valid'][...,None]
    neighbors = torch.cat((g['adjacency'] @ geometry, g['group'] @ geometry), -1)
    flags = torch.stack((g['adjacency'].sum(-1)>0, g['group'].sum(-1)>0), -1).to(state.dtype)
    return torch.cat((own, neighbors, flags), -1) * g['valid'][...,None]

class NodeBlock(GeometryBlock):
    def __init__(self, width, heads=4, ffn_dropout=0., residual_dropout=0.):
        super().__init__(width, heads, ffn_dropout, residual_dropout)
        del self.bias
        self.geometry_q = nn.Linear(26, width)
        self.geometry_k = nn.Linear(26, width)
        self.geometry_scale = nn.Parameter(torch.full((heads,), .1))

    def forward(self, h, g, state):
        b,n,w = h.shape
        q,k,v = self.qkv(self.norm1(h)).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        phi = node_geometry(state, g)
        scale = self.geometry_scale[None,:,None,None]
        q = q + scale*self.geometry_q(phi).view(b,n,self.heads,self.dim).transpose(1,2)
        k = k + scale*self.geometry_k(phi).view(b,n,self.heads,self.dim).transpose(1,2)
        scores = (q @ k.transpose(-2,-1))/math.sqrt(self.dim)
        scores = scores.masked_fill(~g['valid'][:,None,None,:], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1,2).reshape(b,n,w)
        local = self.local(torch.cat((g['adjacency'] @ h, g['group'] @ h), -1))
        h = h+self.residual_dropout(self.output(attended)+local)
        return (h+self.residual_dropout(self.feed_forward(self.norm2(h))))*g['valid'][...,None]

class GeometryModel(OriginalModel):
    def __init__(self, width=64, layers=12, rectangle=True, use_gap=True,
                 ffn_dropout=.1, residual_dropout=.05):
        if not rectangle or not use_gap:
            raise ValueError('Node26 has one registered geometry configuration')
        super().__init__(width,layers,rectangle,use_gap,ffn_dropout,residual_dropout)
        for i, original in enumerate(self.layers):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261109117+i)
                block = NodeBlock(width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
            common = {k:v for k,v in original.state_dict().items() if not k.startswith('bias.')}
            missing, unexpected = block.load_state_dict(common, strict=False)
            assert set(missing)=={'geometry_q.weight','geometry_q.bias','geometry_k.weight','geometry_k.bias','geometry_scale'}
            assert not unexpected
            self.layers[i] = block
