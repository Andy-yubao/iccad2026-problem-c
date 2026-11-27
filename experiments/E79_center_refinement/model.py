"""Frozen coherent E78 plus zero-init center-only correction, source anchor output."""
import torch
from torch import nn
from .warm_model import GeometryModel as Source
from .center_corrector import CenterCorrector
from .base_model import decode
from .frame import contain
from .mib_projection import coherent_shapes

class GeometryModel(nn.Module):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=96,message_width=32,max_correction=.15):
        super().__init__();self.width=width;self.depth=layers;self.max_correction=max_correction
        self.source=Source(width,layers,rectangle,ffn_dropout,corrector_width,message_width,max_correction)
        self.source.requires_grad_(False);self.source.eval()
        self.corrector=CenterCorrector(corrector_width,message_width,ffn_dropout)

    def train(self,mode=True):
        super().train(mode);self.source.eval();return self

    @torch.no_grad()
    def anchor(self,g):
        # Reconstruct the exact E78 forward, reusing its single E74 backbone
        # evaluation and frame. No second expensive source call is needed.
        anchor,frame=self.source.anchor(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.source.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        residual=self.source.corrector(anchor,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        candidate=contain(anchor+residual,g,frame)
        return coherent_shapes(candidate,g,frame),frame

    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}
        if set(g)-allowed:raise ValueError('Only public graph inputs may enter the network')
        source,frame=self.anchor(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        residual=self.corrector(source,g).tanh()*self.max_correction*local[...,None]*g['mask'][...,:2]*g['valid'][...,None]
        wh=decode(source,g)[...,2:]
        centers=torch.minimum(torch.maximum(source[...,:2]+residual,frame[0][:,None]+wh/2),frame[1][:,None]-wh/2)
        centers=torch.where(g['mask'][...,:2].bool(),centers,g['known'][...,:2])
        prediction=torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]
        return torch.cat((prediction,source),-1)
