"""Small frozen placer plus independently warm-widened, trainable correction F."""
import torch
from torch import nn
from .base_model import GeometryModel as Placer,decode
from .old_corrector import Corrector
from .old_center_corrector import CenterCorrector
from .frame import predicted_frame,contain
from .mib_projection import coherent_shapes

PUBLIC={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}

class CorrectionModel(nn.Module):
    def __init__(self,width=192,message_width=64,dropout=.05,max_correction=.15):
        super().__init__();self.max_correction=max_correction
        self.first=Corrector(width,message_width,dropout)
        self.second=Corrector(width,message_width,dropout)
        self.third=CenterCorrector(width,message_width,dropout)

    def forward(self,base,g):
        if set(g)-PUBLIC:raise ValueError('Only public geometry and graph may enter F')
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        old=base+self.first(base,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        frame=predicted_frame(base,g)
        anchor=contain(old,g,frame)
        residual=self.second(anchor,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        candidate=contain(anchor+residual,g,frame)
        source=coherent_shapes(candidate,g,frame)
        residual=self.third(source,g).tanh()*self.max_correction*local[...,None]*g['mask'][...,:2]*g['valid'][...,None]
        wh=decode(source,g)[...,2:]
        centers=torch.minimum(torch.maximum(source[...,:2]+residual,frame[0][:,None]+wh/2),frame[1][:,None]-wh/2)
        centers=torch.where(g['mask'][...,:2].bool(),centers,g['known'][...,:2])
        return torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]

class DeploymentModel(nn.Module):
    """The actual deployment contains P42 and F, with no P74 evaluation."""
    def __init__(self,width=96,layers=12,rectangle=True,ffn_dropout=.05,
                 corrector_width=192,message_width=64,max_correction=.15,alternative_width=192):
        super().__init__();self.width=width;self.depth=layers;self.max_correction=max_correction
        self.placer=Placer(width=width,layers=layers,rectangle=rectangle,ffn_dropout=ffn_dropout)
        self.refiner=CorrectionModel(corrector_width,message_width,ffn_dropout,max_correction)
        self.placer.requires_grad_(False);self.placer.eval()

    def train(self,mode=True):
        super().train(mode);self.placer.eval();return self

    def forward(self,g):
        if set(g)-PUBLIC:raise ValueError('Only public inputs may enter placer and F')
        with torch.no_grad():base=self.placer(g)
        return self.refiner(base,g)

class GeometryModel(DeploymentModel):
    """Train the same F on BOTH sources for every example, not different splits."""
    def __init__(self,width=96,layers=12,rectangle=True,ffn_dropout=.05,
                 corrector_width=192,message_width=64,max_correction=.15,alternative_width=192):
        super().__init__(width,layers,rectangle,ffn_dropout,corrector_width,message_width,max_correction,alternative_width)
        self.alternative=Placer(width=alternative_width,layers=layers,rectangle=rectangle,ffn_dropout=ffn_dropout)
        self.alternative.requires_grad_(False);self.alternative.eval()

    def train(self,mode=True):
        super().train(mode);self.alternative.eval();return self

    def forward(self,g):
        if set(g)-PUBLIC:raise ValueError('Only public inputs may enter placer and F')
        with torch.no_grad():first=self.placer(g);second=self.alternative(g)
        return torch.cat((self.refiner(first,g),self.refiner(second,g)),-1)

def deployment_weights(state):
    return {k:v for k,v in state.items() if not k.startswith('alternative.')}
