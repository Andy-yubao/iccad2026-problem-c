"""Public-context latent replanning alternates with rectangle refinement."""
import math
import torch
from torch import nn
from .original_model import GeometryModel as OriginalModel
from .original_model import decode

class InitialGraphBlock(nn.Module):
    """Geometry-free pre-norm attention and explicit wire/group graph messages."""
    def __init__(self,width=64,heads=4,ffn_dropout=.1,residual_dropout=.05):
        super().__init__()
        self.heads=heads;self.dim=width//heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3);self.output=nn.Linear(width,width)
        self.local=nn.Linear(width*2,width)
        self.ff=nn.Sequential(nn.Linear(width,width*3),nn.SiLU(),nn.Linear(width*3,width))
        self.ff_dropout=nn.Dropout(ffn_dropout);self.residual_dropout=nn.Dropout(residual_dropout)

    def forward(self,h,g):
        b,n,w=h.shape;normalized=self.norm1(h)
        q,k,v=self.qkv(normalized).view(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,w)
        local=self.local(torch.cat((g['adjacency']@normalized,g['group']@normalized),-1))
        h=h+self.residual_dropout(self.output(attended)+local)
        ff=self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](self.norm2(h)))))
        return (h+self.residual_dropout(ff))*g['valid'][...,None]

class GeometryModel(OriginalModel):
    def __init__(self,width=64,layers=12,rectangle=True,use_gap=True,
                 ffn_dropout=.1,residual_dropout=.05,initial_layers=2):
        if initial_layers!=2:raise ValueError('Only the registered two initial graph blocks are authorized')
        super().__init__(width,layers,rectangle,use_gap,ffn_dropout,residual_dropout)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261109119)
            self.initial_blocks=nn.ModuleList([InitialGraphBlock(width,ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout) for _ in range(initial_layers)])

    def planning_block(self,block,h,g):
        """Unrestricted latent attention; pair bias contains public context only."""
        b,n,w=h.shape
        q,k,v=block.qkv(block.norm1(h)).view(b,n,3,block.heads,block.dim).permute(2,0,3,1,4).unbind(0)
        local=g['area'].clamp_min(1e-12).sqrt()
        locked=(g['constraints'][...,:2]!=0).any(-1)
        wh=torch.where(locked[...,None],g['specs'][...,2:]/local[...,None],torch.ones_like(g['specs'][...,2:]))
        dims=torch.cat((wh[:,:,None].expand(-1,-1,n,-1),wh[:,None,:].expand(-1,n,-1,-1)),-1)
        flags=g['constraints'][...,:2].to(h.dtype)
        anchors=torch.cat((flags[:,:,None].expand(-1,-1,n,-1),flags[:,None,:].expand(-1,n,-1,-1)),-1)
        zero=h.new_zeros((b,n,n,4))
        pair=torch.cat((zero,g['adjacency'][...,None],(g['group']>0)[...,None],dims,zero,anchors),-1)
        scores=(q@k.transpose(-2,-1))/math.sqrt(block.dim)+block.bias(pair).permute(0,3,1,2)
        scores=scores.masked_fill(~g['valid'][:,None,None,:],-1e4)
        attended=(scores.softmax(-1)@v).transpose(1,2).reshape(b,n,w)
        message=block.local(torch.cat((g['adjacency']@h,g['group']@h),-1))
        h=h+block.residual_dropout(block.output(attended)+message)
        return (h+block.residual_dropout(block.feed_forward(block.norm2(h))))*g['valid'][...,None]

    def corruption_pattern(self,g):
        """Public-only pairing, random in training and fixed during monitoring."""
        eligible=g['valid']&(g['constraints'][...,1]==0)
        b,n=eligible.shape;index=torch.arange(n,device=eligible.device)[None,:].expand(b,-1)
        if self.training:
            score=torch.rand((b,n),device=eligible.device)
            stage=torch.randint(0,3,(b,),device=eligible.device)
        else:
            score=torch.frac(torch.sin((index+1)*12.9898+g['features'][...,0]*78.233)*43758.5453).abs()
            stage=g['valid'].sum(-1)%3
        order=score.masked_fill(~eligible,2.).argsort(dim=-1,stable=True)
        # Inverse permutation without deterministic scatter's host-side checks.
        rank=order.argsort(dim=-1,stable=True)
        count=eligible.sum(-1)
        amount=torch.where(count>=2,((count//8)*2).clamp_min(2),torch.zeros_like(count))
        partner_rank=torch.where(rank<amount[:,None],amount[:,None]-1-rank,rank).clamp_min(0)
        partner=order.gather(1,partner_rank)
        return partner,stage

    def forward(self,g,corrupt=False):
        if set(g)-{'features','adjacency','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph inputs may enter the network')
        h=self.input(g['features'])*g['valid'][...,None]
        for block in self.initial_blocks:h=block(h,g)
        context=h
        state=self.head(h)
        states=[state]  # Observe the existing S0; no extra head call.
        pattern=self.corruption_pattern(g) if corrupt else None
        for index,layer in enumerate(self.layers,1):
            if index in (4,7,10):
                if corrupt:
                    partner,stage=pattern
                    swapped=state[...,:2].detach().gather(1,partner[...,None].expand(-1,-1,2))
                    spoiled=torch.cat((swapped,state[...,2:].detach()),-1)
                    active=(stage==(index-4)//3)[:,None,None]
                    state=torch.where(active,spoiled,state)
                # No clean geometry-conditioned hidden state bypasses this reset.
                h=self.planning_block(layer,context+self.state_embed(state*g['mask']),g)
            elif index in (5,8,11):
                # The previous planner readout is observable, never fed back here.
                h=self.planning_block(layer,h,g)
            else:
                h=layer(h+self.state_embed(state*g['mask']),g,state)
            state=self.head(h);states.append(state)
        return torch.stack(states,dim=0)
