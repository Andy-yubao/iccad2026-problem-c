"""Warm E77 last corrector with public MIB-coherent output and same frame."""
import torch
from .pretrained_model import GeometryModel as Base
from .base_model import decode
from .frame import contain
from .mib_projection import coherent_shapes

class GeometryModel(Base):
    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}
        if set(g)-allowed:raise ValueError('Only public graph inputs may enter the network')
        anchor,frame=self.anchor(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        residual=self.corrector(anchor,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        candidate=contain(anchor+residual,g,frame)
        return coherent_shapes(candidate,g,frame)
