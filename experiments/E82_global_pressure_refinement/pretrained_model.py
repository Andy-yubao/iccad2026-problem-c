"""Frozen E76 plus trainable internal correction in E74's predicted frame."""
import torch
from torch import nn
from .source_model import GeometryModel as Source
from .base_model import decode
from .corrector import Corrector
from .frame import predicted_frame, contain


class GeometryModel(nn.Module):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,
                 corrector_width=96,message_width=32,max_correction=.15):
        super().__init__()
        self.width=width;self.depth=layers;self.max_correction=max_correction
        self.source=Source(width,layers,rectangle,ffn_dropout,corrector_width,message_width,max_correction)
        self.source.requires_grad_(False);self.source.eval()
        self.corrector=Corrector(corrector_width,message_width,ffn_dropout)

    def train(self,mode=True):
        super().train(mode);self.source.eval();return self

    @torch.no_grad()
    def anchor(self,g):
        # Compute the expensive E74 backbone exactly once. This is the same
        # original E76 function, followed by the new explicit frame projection.
        base=self.source.backbone(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.source.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        old=base+self.source.corrector(base,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        frame=predicted_frame(base,g)
        return contain(old,g,frame),frame

    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}
        if set(g)-allowed:raise ValueError('Only public graph inputs may enter the network')
        anchor,frame=self.anchor(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        residual=self.corrector(anchor,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        return contain(anchor+residual,g,frame)
