"""Frozen deployment interventions; copied native E81 retained for parity checks."""
import torch
from .native_model import DeploymentModel,decode
from .frame import predicted_frame,contain
from .mib_projection import coherent_shapes

class GeometryModel(DeploymentModel):
    def __init__(self,learned_residuals=True,**kwargs):
        super().__init__(**kwargs);self.learned_residuals=learned_residuals
        self.requires_grad_(False);self.eval()

    def train(self,mode=True):return super().train(False)

    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}
        if set(g)-allowed:raise ValueError('Only public inputs may enter this experiment')
        base=self.placer(g)
        if self.learned_residuals:return self.refiner(base,g)
        # Exactly preserve the original operator ordering while bypassing all
        # three learned residual branches. No calls to any F weights occur.
        frame=predicted_frame(base,g)
        anchor=contain(base,g,frame)
        candidate=contain(anchor,g,frame)
        source=coherent_shapes(candidate,g,frame)
        wh=decode(source,g)[...,2:]
        centers=torch.minimum(torch.maximum(source[...,:2],frame[0][:,None]+wh/2),frame[1][:,None]-wh/2)
        centers=torch.where(g['mask'][...,:2].bool(),centers,g['known'][...,:2])
        return torch.cat((centers,source[...,2:3]),-1)*g['valid'][...,None]
