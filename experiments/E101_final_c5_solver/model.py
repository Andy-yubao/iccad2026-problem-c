"""Small hybrid student with trainable unbounded proposals, public-only maps."""
import torch
from .nn.model import CorrectionModel as OriginalCorrection,DeploymentModel as OriginalDeployment
from .nn.base_model import decode
from .nn.frame import predicted_frame
from .nn.mib_projection import coherent_shapes
from .nn.cooperation import spatial_centers

def canonical(state,g):
    p=decode(state,g)
    center=torch.maximum(state[...,:2],p[...,2:]/2)
    result=torch.cat((center,state[...,2:3].clamp(-5,5)),-1)
    return torch.where(g['mask'].bool(),result,g['known'])*g['valid'][...,None]

class CorrectionModel(OriginalCorrection):
    def forward(self,base,g):
        local=g['area'].clamp_min(1e-12).sqrt();unit=torch.stack((local,local,torch.ones_like(local)),-1)
        # Linear residual scale controls parameter units; it is not a hard cap.
        state=canonical(base+.5*self.first(base,g)*unit*g['mask'],g)
        state=canonical(state+.5*self.second(state,g)*unit*g['mask'],g)
        frame=predicted_frame(state,g);source=coherent_shapes(state,g,frame)
        residual=self.third(source,g).tanh()*.15*local[...,None]*g['mask'][...,:2]
        proposed=source[...,:2]+residual
        centers=spatial_centers(source,proposed,g,frame,.15)
        return torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]

class DeploymentModel(OriginalDeployment):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=192,message_width=64,max_correction=.15,cooperative=True):
        super().__init__(width,layers,rectangle,ffn_dropout,corrector_width,message_width,max_correction,cooperative)
        self.refiner=CorrectionModel(corrector_width,message_width,ffn_dropout,max_correction,cooperative)
