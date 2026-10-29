"""One frozen placer and exactly the same three public-geometry correctors."""
import torch
from torch import nn
from .base_model import GeometryModel as Placer,decode
from .corrector import Corrector
from .center_corrector import CenterCorrector
from .frame import predicted_frame,contain
from .mib_projection import coherent_shapes

class GeometryModel(nn.Module):
    def __init__(self,width=96,layers=12,rectangle=True,ffn_dropout=.05,
                 corrector_width=96,message_width=32,max_correction=.15):
        super().__init__();self.width=width;self.depth=layers;self.max_correction=max_correction
        self.placer=Placer(width=width,layers=layers,rectangle=rectangle,ffn_dropout=ffn_dropout)
        self.first=Corrector(corrector_width,message_width,ffn_dropout)
        self.second=Corrector(corrector_width,message_width,ffn_dropout)
        self.third=CenterCorrector(corrector_width,message_width,ffn_dropout)
        self.requires_grad_(False);self.eval()

    def train(self,mode=True):
        # This experiment performs no optimization, including accidental dropout.
        return super().train(False)

    def anchor(self,g):
        base=self.placer(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        bounds=self.max_correction*torch.stack((local,local,torch.ones_like(local)),-1)
        old=base+self.first(base,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        frame=predicted_frame(base,g)
        anchor=contain(old,g,frame)
        residual=self.second(anchor,g).tanh()*bounds*g['mask']*g['valid'][...,None]
        candidate=contain(anchor+residual,g,frame)
        return coherent_shapes(candidate,g,frame),frame

    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}
        if set(g)-allowed:raise ValueError('Only public graph inputs may enter the network')
        source,frame=self.anchor(g)
        local=g['area'].clamp_min(1e-12).sqrt()
        residual=self.third(source,g).tanh()*self.max_correction*local[...,None]*g['mask'][...,:2]*g['valid'][...,None]
        wh=decode(source,g)[...,2:]
        centers=torch.minimum(torch.maximum(source[...,:2]+residual,frame[0][:,None]+wh/2),frame[1][:,None]-wh/2)
        centers=torch.where(g['mask'][...,:2].bool(),centers,g['known'][...,:2])
        prediction=torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]
        return torch.cat((prediction,source),-1)

def extract(model,source,placer=None):
    """Strict mapping; F has no width-dependent hidden placer features."""
    pairs=((model.first,'source.source.corrector.'),(model.second,'source.corrector.'),(model.third,'corrector.'))
    for module,prefix in pairs:
        module.load_state_dict({k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)},strict=True)
    if placer is None:
        prefix='source.source.backbone.'
        placer={k[len(prefix):]:v for k,v in source.items() if k.startswith(prefix)}
    model.placer.load_state_dict(placer,strict=True)
    return model
