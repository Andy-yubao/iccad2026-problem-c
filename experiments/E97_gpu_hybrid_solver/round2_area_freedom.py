"""Proposal module: bounded public area-tolerance head, no private NN inputs.

Not a registered training candidate by itself. Source selection/plan/preflight
must be completed before using it in a new run.
"""
import torch
from torch import nn
from .student import DeploymentModel as StudentDeployment
from .nn85.base_model import decode as nominal_decode

def decode(state,g):
    if state.shape[-1]==3:return nominal_decode(state,g)
    return nominal_decode(state[...,:3],dict(g,area=g['area']*(1+state[...,3])))

class AreaRefiner(nn.Module):
    def __init__(self,draft,width=64):
        super().__init__();self.draft=draft;self.area_head=nn.Sequential(nn.Linear(22,width),nn.SiLU(),nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.area_head[-1].weight);nn.init.zeros_(self.area_head[-1].bias)
    def forward(self,base,g):
        state=self.draft(base,g);local=g['area'].clamp_min(1e-12).sqrt();wh=nominal_decode(state,g)[...,2:]
        shape=(wh/local[...,None]).clamp_min(1e-6).log().clamp(-5,5)
        raw=self.area_head(torch.cat((g['features'],state,shape),-1))[...,0]
        group=g['constraints'][...,2];same=(group[:,:,None]==group[:,None,:])&(group[:,:,None]>0)&g['valid'][:,:,None]&g['valid'][:,None,:]
        count=same.sum(-1);mean=(same.to(raw.dtype)*raw[:,None,:]).sum(-1)/count.clamp_min(1)
        raw=torch.where(count>0,mean,raw)
        locked=((~g['mask'][...,2].bool())&g['valid']);anchored=(same&locked[:,None,:]).any(-1)
        fraction=.00999*raw.tanh()*g['mask'][...,2]*g['valid']*(~anchored)
        return torch.cat((state,fraction[...,None]),-1)

class DeploymentModel(StudentDeployment):
    def __init__(self,area_width=64,**kwargs):
        super().__init__(**kwargs);self.refiner=AreaRefiner(self.refiner,area_width)
