"""E10 attention initialization plus optional supervised structure heads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from .backend.frozen.model import GeometryModel
from .io import ROOT,sha

CHECKPOINT=ROOT/'experiments/E10_integrated_baseline/runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'
CHECKPOINT_SHA256='47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28'

class Student(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=GeometryModel('attention',96,4)
        self.boundary=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,4))
        self.pair=nn.Sequential(nn.Linear(96*2+6,64),nn.SiLU(),nn.Linear(64,6))
    def initialize(self):
        if sha(CHECKPOINT)!=CHECKPOINT_SHA256:raise ValueError('Frozen E10 checkpoint changed')
        state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        self.backbone.load_state_dict(state['model'],strict=True)
    def forward(self,g,structure=True):
        if 'target' in g or 'reference' in g:raise ValueError('Supervision in model input')
        b=self.backbone;h=b.input(g['features'])*g['valid'][...,None];state=b.head(h)
        for layer in b.layers:
            h=layer(h+b.state_embed(state*g['mask']),g,state);state=b.head(h)
        state=torch.where(g['mask'].bool(),state,g['known'])
        out={'geometry':state}
        if structure:
            n=h.shape[1]
            pair=torch.cat((h[:,:,None,:]+h[:,None,:,:],h[:,:,None,:]-h[:,None,:,:],
              state[:,:,None,:]-state[:,None,:,:],g['adjacency'][...,None],
              (g['mib']>0)[...,None],(g['group']>0)[...,None]),-1)
            logits=self.pair(pair);out.update(boundary=self.boundary(h),contact=logits[...,:5],mib=logits[...,5])
        return out
