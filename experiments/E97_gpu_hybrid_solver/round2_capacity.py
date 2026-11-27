"""C3 small F plus normalized global adapter, jointly trainable, initial identity."""
from torch import nn
from .normalized_geometry import DeploymentModel as NormalizedDeployment

class JointRefiner(nn.Module):
    def __init__(self,draft,adapter):
        super().__init__();self.draft=draft;self.adapter=adapter
    def forward(self,base,g):return self.adapter(self.draft(base,g),g)

class DeploymentModel(NormalizedDeployment):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.refiner=JointRefiner(self.refiner.draft,self.refiner.adapter)
