"""Frozen E74 backbone plus zero-initialized, bounded geometry corrector."""
import torch
from torch import nn
from .base_model import GeometryModel as Backbone,decode
from .corrector import Corrector
class GeometryModel(nn.Module):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=96,message_width=32,max_correction=.15):
        super().__init__();self.width=width;self.depth=layers;self.max_correction=max_correction
        self.backbone=Backbone(width=width,layers=layers,rectangle=rectangle,ffn_dropout=ffn_dropout)
        self.backbone.requires_grad_(False);self.backbone.eval()
        self.corrector=Corrector(corrector_width,message_width,ffn_dropout)
    @property
    def layers(self):return self.backbone.layers
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}
        if set(g)-allowed:raise ValueError('Only public graph inputs may enter the network')
        with torch.no_grad():state=self.backbone(g)
        raw=self.corrector(state,g)
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        return state+raw.tanh()*bounds*g['mask']*g['valid'][...,None]
