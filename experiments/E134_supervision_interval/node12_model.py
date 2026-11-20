"""Approved Node12 collision statistics in Q/K, with the complete E119 generator."""
import math
import torch
from torch import nn
from .e119_model import GeometryModel as E119Model
from .original_model import GeometryBlock,decode

FEATURE_NAMES=('cx','cy','w','h','overlap_count_fraction','log_overlap_load',
               'max_overlap_fraction','log_max_separation','log_left_pressure',
               'log_right_pressure','log_down_pressure','log_up_pressure')
AXIS_SWAP=(1,0,3,2,4,5,6,7,10,11,8,9)
TEMPERATURE=.25

def node_geometry(state,g):
    """All continuous statistics backpropagate; booleans carry no surrogate gradient."""
    rectangles=decode(state,g)
    left,bottom=rectangles[...,0],rectangles[...,1]
    width,height=rectangles[...,2],rectangles[...,3]
    right=left+width;top=bottom+height
    valid=g['valid'];n=state.shape[1]
    pair_valid=(valid[:,:,None]&valid[:,None,:])&~torch.eye(n,dtype=torch.bool,device=state.device)[None]
    ix=(torch.minimum(right[:,:,None],right[:,None,:])-torch.maximum(left[:,:,None],left[:,None,:])).clamp_min(0)
    iy=(torch.minimum(top[:,:,None],top[:,None,:])-torch.maximum(bottom[:,:,None],bottom[:,None,:])).clamp_min(0)
    overlapping=(ix>0)&(iy>0)&pair_valid
    intersection=ix*iy*pair_valid.to(state.dtype)
    area=g['area'].clamp_min(1e-12)
    overlap_fraction=intersection/area[:,:,None]
    local=area.sqrt()
    scale=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
    # Distances to move i left/right/down/up until it no longer intersects j.
    distances=torch.stack((right[:,:,None]-left[:,None,:],
                           right[:,None,:]-left[:,:,None],
                           top[:,:,None]-bottom[:,None,:],
                           top[:,None,:]-bottom[:,:,None]),-1)/scale[...,None]
    shortest=distances.amin(-1).clamp_min(0)*overlapping.to(state.dtype)
    direction=(-distances/TEMPERATURE).softmax(-1)
    pressure=(shortest[...,None]*direction).sum(-2)
    count=overlapping.to(state.dtype).sum(-1)/(valid.sum(-1)-1).clamp_min(1).to(state.dtype)[:,None]
    summary=torch.stack((count,overlap_fraction.sum(-1).log1p(),
                         overlap_fraction.amax(-1),shortest.amax(-1).log1p()),-1)
    own=torch.stack((left+width/2,bottom+height/2,width,height),-1)
    return torch.cat((own,summary,pressure.log1p()),-1)*valid[...,None]

class Node12Block(GeometryBlock):
    def __init__(self,width,heads=4,ffn_dropout=.1,residual_dropout=.05):
        super().__init__(width,heads,ffn_dropout,residual_dropout)
        del self.bias
        self.geometry_q=nn.Linear(12,width)
        self.geometry_k=nn.Linear(12,width)
        self.geometry_scale=nn.Parameter(torch.full((heads,),.1))

    def forward(self,h,g,state):
        b,n,w=h.shape
        q,k,v=self.qkv(self.norm1(h)).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        phi=node_geometry(state,g)
        scale=self.geometry_scale[None,:,None,None]
        q=q+scale*self.geometry_q(phi).view(b,n,self.heads,self.dim).transpose(1,2)
        k=k+scale*self.geometry_k(phi).view(b,n,self.heads,self.dim).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,w)
        aggregated=self.local(torch.cat((g['adjacency']@h,g['group']@h),-1))
        h=h+self.residual_dropout(self.output(attended)+aggregated)
        return (h+self.residual_dropout(self.feed_forward(self.norm2(h))))*g['valid'][...,None]

class GeometryModel(E119Model):
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,
                 ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        if not rectangle or not use_gap:raise ValueError('Only the registered Node12 geometry configuration is authorized')
        super().__init__(width,layers,rectangle,use_gap,ffn_dropout,residual_dropout,initial_layers)
        for i,original in enumerate(self.layers):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261109120+i)
                layer=Node12Block(width,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout)
            common={k:v for k,v in original.state_dict().items() if not k.startswith('bias.')}
            missing,unexpected=layer.load_state_dict(common,strict=False)
            assert set(missing)=={'geometry_q.weight','geometry_q.bias','geometry_k.weight','geometry_k.bias','geometry_scale'}
            assert not unexpected
            self.layers[i]=layer
