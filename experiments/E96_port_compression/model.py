"""Independent deep neural encoder with ONE simultaneous module readout.

All layers exchange hidden features. Draft geometry is encoded once and never
updated inside the encoder. No descent, line search, containment repair, shape
averaging, solver, recurrent coordinate head, or per-module decoding loop.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel,SDPBackend
from .base_model import GeometryModel as Placer,RectangleBlock,decode
from .data import PUBLIC,placer_inputs
from .compression import network_view

def canonical_state(state,g):
    """Represent the actually decoded rectangles, including public locks/clamps."""
    p=decode(state,g)
    # Clamp the existing center/ratio directly. A decode/re-encode round trip
    # introduces tiny repeated changes even when the neural residual is zero.
    value=torch.cat((torch.maximum(state[...,:2],p[...,2:]/2),
        state[...,2:].clamp(-5,5)),-1)
    return torch.where(g['mask'].bool(),value,g['known'])*g['valid'][...,None]

class GlobalBlock(RectangleBlock):
    def attend(self,q,k,v,mm,port_bias,token_valid):
        b,_,n,_=mm.shape;p=port_bias.shape[-1]
        upper=torch.cat((mm,port_bias),-1)
        lower=torch.cat((port_bias.transpose(-1,-2),q.new_zeros(b,self.heads,p,p)),-1)
        bias=torch.cat((upper,lower),-2)
        if self.attention_backend=='efficient':
            # Explicit backend: never silently fall back to quadratic math SDPA.
            mask=bias.masked_fill(~token_valid[:,None,None,:],float('-inf'))
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dim)+bias
        scores=scores.masked_fill(~token_valid[:,None,None,:],-1e4)
        return scores.softmax(-1)@v

    def forward(self,h,g,pairs,port_bias,token_valid,raw_bias):
        b,t,w=h.shape;n=g['valid'].shape[1];p=t-n
        normalized=self.norm1(h)
        q,k,v=self.qkv(normalized).view(b,t,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        mm=self.bias(pairs).permute(0,3,1,2)+raw_bias
        if self.attention_backend=='manual' and self.training and torch.is_grad_enabled() and t>448:
            # Only large batches recompute the deterministic attention core.
            # Avoid retaining twelve quadratic softmax/bias matrices in16GiB.
            attended=checkpoint(self.attend,q,k,v,mm,port_bias,token_valid,use_reentrant=False,preserve_rng_state=False)
        else:attended=self.attend(q,k,v,mm,port_bias,token_valid)
        attended=attended.transpose(1,2).reshape(b,t,w)
        # The old unnormalised local path was a repeated learned linear
        # feedback operator: its hidden RMS grew from9 to120k in the failed run.
        # Use the same pre-normalisation as attention before graph aggregation.
        nodes=normalized[:,:n] if self.normalize_local else h[:,:n]
        local=self.local(torch.cat((g['adjacency']@nodes,g['mib']@nodes,g['group']@nodes),-1))
        local=torch.cat((local,torch.zeros_like(h[:,n:])),1)
        h=h+self.output(attended)+local
        return (h+self.feed_forward(self.norm2(h)))*token_valid[...,None]

class GlobalCorrector(nn.Module):
    """Public problem + canonical draft state -> all corrected module states."""
    def __init__(self,width=192,layers=12,heads=4,dropout=.05,normalize_local=True,attention_backend='manual',port_mode='centroid'):
        super().__init__();self.width=width;self.depth=layers;self.heads=heads
        self.port_mode=port_mode
        self.input=nn.Sequential(nn.Linear(17,width),nn.SiLU(),nn.LayerNorm(width))
        self.state_embed=nn.Linear(3,width)
        self.port_input=nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.LayerNorm(width))
        self.port_pair=nn.Sequential(nn.Linear(5,24),nn.SiLU(),nn.Linear(24,heads))
        self.weight_bias=nn.Linear(1,heads,bias=False)
        nn.init.zeros_(self.weight_bias.weight)
        self.layers=nn.ModuleList([GlobalBlock(width,heads=heads,ffn_dropout=dropout) for _ in range(layers)])
        if attention_backend not in ('manual','efficient'):raise ValueError('Unregistered attention backend')
        for layer in self.layers:
            layer.normalize_local=normalize_local;layer.attention_backend=attention_backend
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def geometry_features(self,draft,g):
        p=decode(draft,g);wh=p[...,2:];center=p[...,:2]+wh/2;n=draft.shape[1]
        d=draft[:,:,:2][:,:,None]-draft[:,:,:2][:,None,:]
        local=g['area'].clamp_min(1e-12).sqrt();scale=(local[:,:,None]+local[:,None,:])/2
        gap=((center[:,:,None]-center[:,None,:]).abs()-(wh[:,:,None]+wh[:,None,:])/2)/scale.clamp_min(1e-6)[...,None]
        shape=wh/local[...,None];flags=g['constraints'][...,:2].to(draft.dtype)
        pair=torch.cat((d,d.abs(),g['adjacency'][...,None],(g['mib']>0)[...,None],(g['group']>0)[...,None],
            shape[:,:,None].expand(-1,-1,n,-1),shape[:,None,:].expand(-1,n,-1,-1),gap,(-gap).clamp_min(0),
            flags[:,:,None].expand(-1,-1,n,-1),flags[:,None,:].expand(-1,n,-1,-1)),-1)
        delta=center[:,:,None]-g['port_xy'][:,None]
        port=torch.cat((delta,delta.abs(),g['p2b_weights'].log1p()[...,None]),-1)
        return pair,port

    def forward(self,g,draft):
        if set(g)-set(PUBLIC):raise ValueError('Reference labels and nonpublic inputs forbidden')
        if draft.shape!=(*g['valid'].shape,3):raise ValueError('Expected one draft state for every module')
        g=network_view(g,self.port_mode)
        token_valid=torch.cat((g['valid'],g['port_valid']),1)
        draft=canonical_state(draft,g)
        nodes=self.input(g['features'])+self.state_embed(draft)
        ports=self.port_input(torch.cat((g['port_xy'],g['p2b_weights'].sum(1).log1p()[...,None]),-1))
        h=torch.cat((nodes,ports),1)*token_valid[...,None]
        pairs,port_features=self.geometry_features(draft,g)
        port_bias=self.port_pair(port_features).permute(0,3,1,2)
        raw_bias=self.weight_bias(g['b2b_weights'].log1p()[...,None]).permute(0,3,1,2)
        # Full connectivity weights enter every global layer without replacing
        # missing physical edges by fictitious edges.
        for layer in self.layers:
            h=layer(h,g,pairs,port_bias,token_valid,raw_bias)
        valid=token_valid[...,None]
        self.last_hidden_rms=((h.square()*valid).sum((1,2))/(valid.sum((1,2))*self.width).clamp_min(1e-12)).sqrt()
        delta=self.head(h[:,:draft.shape[1]])
        result=draft+delta*g['mask']
        return torch.where(g['mask'].bool(),result,g['known'])*g['valid'][...,None]

class GeometryModel(nn.Module):
    """Convenience deployment wrapper; the correction network is independently exportable."""
    def __init__(self,width=192,layers=12,heads=4,dropout=.05,normalize_local=True,attention_backend='manual',port_mode='centroid'):
        super().__init__()
        self.placer=Placer(width=192,layers=12,rectangle=True,ffn_dropout=.05)
        self.corrector=GlobalCorrector(width,layers,heads,dropout,normalize_local,attention_backend,port_mode)
        self.placer.requires_grad_(False);self.placer.eval()
    def train(self,mode=True):super().train(mode);self.placer.eval();return self
    def draft(self,g):
        with torch.no_grad():return canonical_state(self.placer(placer_inputs(g)),g)
    def forward(self,g):
        if set(g)-set(PUBLIC):raise ValueError('Only public problem information is accepted')
        return self.corrector(g,self.draft(g))
