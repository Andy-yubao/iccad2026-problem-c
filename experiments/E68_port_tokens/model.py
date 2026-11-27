"""Combined full-port representation: no legacy pin summaries or weight-scale loss."""
import torch
from torch import nn
from .base_model import GeometryModel as BaseModel, decode


class PortReader(nn.Module):
    def __init__(self,width,hidden,mode,scale):
        super().__init__();self.mode=mode;self.scale=scale
        self.norm=nn.LayerNorm(width)
        self.receiver=nn.Linear(width,hidden)
        self.geometry=nn.Linear(7,hidden)
        self.score=nn.Linear(hidden,1,bias=False)
        self.output=nn.Linear(hidden,width)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,h,g,state):
        weight=g['pin_weights']*g['pin_valid']
        normalized=weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
        coordinates=g['pin_coordinates']
        if self.mode=='centroid':
            # Same slot count, weights and parameters; all connected coordinates
            # become the block's single weighted centroid. Weighted pooling then
            # contains no extra distribution/count information in exact arithmetic.
            mean=(normalized[...,None]*coordinates).sum(-2)
            coordinates=mean[:,:,None,:].expand_as(coordinates)
        p=decode(state,g);center=p[...,:2]+p[...,2:]/2
        delta=coordinates-center[:,:,None,:]
        geometry=torch.cat((coordinates,delta,delta.abs(),weight.log1p()[...,None]),-1)
        value=torch.nn.functional.silu(self.receiver(self.norm(h))[:,:,None,:]+self.geometry(geometry))
        valid=(weight>0)&g['valid'][:,:,None]
        # A normalized attention average alone cancels uniform rescaling of
        # connection weights. Preserve absolute strengths and multiplicity with
        # weighted gated sums; asinh is a stable, injective magnitude transform.
        gate=self.score(value).sigmoid()*valid[...,None]
        aggregated=(weight[...,None]*gate*value).sum(2).asinh()
        has_pin=valid.any(-1)&g['valid']
        return self.scale*self.output(aggregated)*has_pin[...,None]


class GeometryModel(BaseModel):
    def __init__(self,width=96,layers=12,rectangle=True,ffn_dropout=.05,
                 port_mode='full',port_width=24,port_scale=.1):
        super().__init__(width=width,layers=layers,rectangle=rectangle,ffn_dropout=ffn_dropout)
        if port_mode not in ('full','centroid'):raise ValueError(port_mode)
        self.port_mode=port_mode
        with torch.random.fork_rng(devices=[]):
            old_input=self.input[0]
            self.input[0]=nn.Linear(14,width)
            with torch.no_grad():
                self.input[0].weight.copy_(old_input.weight[:,:14]);self.input[0].bias.copy_(old_input.bias)
            self.port_readers=nn.ModuleList([PortReader(width,port_width,port_mode,port_scale) for _ in range(layers)])

    def forward(self,g):
        allowed={'features','adjacency','mib','group','mask','known','area','specs',
                 'constraints','valid','scale','pin_coordinates','pin_weights','pin_valid'}
        if set(g)-allowed:raise ValueError('Only public graph and port inputs may enter the network')
        if g['features'].shape[-1]!=14:raise ValueError('Legacy pin weight/centroid summaries must not enter the model')
        h=self.input(g['features'])*g['valid'][...,None];state=self.head(h)
        for layer,reader in zip(self.layers,self.port_readers):
            incoming=h+self.state_embed(state*g['mask'])
            # Directly connected nodes read full ports. Original global attention
            # and graph/group messages propagate that information in later layers.
            h=(layer(incoming,g,state)+reader(incoming,g,state))*g['valid'][...,None]
            state=self.head(h)
        return state
