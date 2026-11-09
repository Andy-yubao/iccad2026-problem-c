"""Conditioned four-block geometry adapter; post norms and fresh normalized readout."""
import copy
import torch
from torch import nn
from .student import DeploymentModel as StudentDeployment
from .nn85.base_model import RectangleBlock,decode

class GeometryAdapter(nn.Module):
    def __init__(self,width=192,dropout=.05):
        super().__init__();self.input=nn.Sequential(nn.Linear(22,width),nn.SiLU(),nn.LayerNorm(width))
        self.state_embed=nn.Linear(3,width)
        self.layers=nn.ModuleList([RectangleBlock(width,ffn_dropout=dropout) for _ in range(4)])
        self.post_norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    @torch.no_grad()
    def initialize_from_placer(self,placer):
        self.input[0].weight.zero_();self.input[0].weight[:,:17].copy_(placer.input[0].weight);self.input[0].bias.copy_(placer.input[0].bias)
        self.input[2].load_state_dict(placer.input[2].state_dict());self.state_embed.load_state_dict(placer.state_embed.state_dict())
        self.layers.load_state_dict(nn.ModuleList([copy.deepcopy(v) for v in placer.layers[-4:]]).state_dict(),strict=True)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,base,g):
        local=g['area'].clamp_min(1e-12).sqrt();wh=decode(base,g)[...,2:]
        shape=(wh/local[...,None]).clamp_min(1e-6).log().clamp(-5,5)
        h=self.input(torch.cat((g['features'],base,shape),-1))*g['valid'][...,None]
        state=base;unit=torch.stack((local,local,torch.ones_like(local)),-1)
        for layer,normalization in zip(self.layers,self.post_norms):
            h=normalization(layer(h+self.state_embed(state*g['mask']),g,state))
            state=state+.25*self.head(h)*unit*g['mask']*g['valid'][...,None]
        return state

class DeepRefiner(nn.Module):
    def __init__(self,draft,width,dropout):
        super().__init__();self.draft=draft;self.adapter=GeometryAdapter(width,dropout)
    def train(self,mode=True):super().train(mode);self.draft.eval();return self
    def forward(self,base,g):
        with torch.no_grad():state=self.draft(base,g)
        return self.adapter(state,g)

class DeploymentModel(StudentDeployment):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=192,message_width=64,max_correction=.15,cooperative=True):
        super().__init__(width,layers,rectangle,ffn_dropout,corrector_width,message_width,max_correction,cooperative)
        self.refiner=DeepRefiner(self.refiner,width,ffn_dropout)
