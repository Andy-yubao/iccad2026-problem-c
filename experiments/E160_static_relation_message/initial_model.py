"""Three attention/FFN blocks, one final initial-layout decode with public structural/anchor roles."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .public import PUBLIC

class RoleBlock(nn.Module):
    def __init__(self,width=128,heads=4,ffn_dropout=.1,residual_dropout=.05):
        super().__init__();self.heads=heads;self.dim=width//heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.output=nn.Linear(width,width);self.local=nn.Linear(2*width,width)
        self.bias=nn.Sequential(nn.Linear(7,24),nn.SiLU(),nn.Linear(24,heads))
        self.mib_local=nn.Linear(width,width,bias=False)
        self.ff=nn.Sequential(nn.Linear(width,3*width),nn.SiLU(),nn.Linear(3*width,width))
        self.ff_dropout=nn.Dropout(ffn_dropout);self.residual_dropout=nn.Dropout(residual_dropout)
        # Isolate new CPU initialization draws so the E159 common branches keep
        # their original scratch RNG sequence. Each attention block owns psi.
        self.relation_rank=8
        with torch.random.fork_rng(devices=[]):
            self.relation_receiver=nn.Linear(width,self.relation_rank,bias=False)
            self.relation_sender=nn.Linear(width,self.relation_rank,bias=False)
            self.relation_edges=nn.Linear(9,self.relation_rank)
            self.relation_value=nn.ModuleList([nn.Linear(self.relation_rank,self.dim,bias=False) for _ in range(self.heads)])
            with torch.no_grad():
                for projection in self.relation_value:projection.weight.mul_(.01)

    def forward(self,h,ports,valid,pair,adjacency,group,mib):
        b,n,w=h.shape;t=n+ports.shape[1];z=self.norm1(torch.cat((h,ports),1))
        q,k,v=self.qkv(z).view(b,t,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        scores=(q[:,:,:n]@k.transpose(-2,-1))/math.sqrt(self.dim)+self.bias(pair).permute(0,3,1,2)
        attention=scores.masked_fill(~valid[:,None,None,:],-1e4).softmax(-1)
        receiver=self.relation_receiver(z[:,:n]).tanh()
        sender=self.relation_sender(z).tanh()
        is_module=torch.arange(t,device=z.device)<n
        sender_types=torch.stack((is_module,~is_module),-1).to(z.dtype)
        # The nine-dimensional static e is pair7 plus sender type2. Splitting
        # its linear projection avoids allocating a dense B*N*T*9 copy.
        edge_message=(F.linear(pair,self.relation_edges.weight[:,:7],self.relation_edges.bias)
            +F.linear(sender_types,self.relation_edges.weight[:,7:])[None,None,:,:]).tanh()
        # psi_ij^a=U_a[r_i*(s_j+f(e_ij))]. Pull r_i outside sum_j so
        # receiver-conditioned messages do not require full-width pair values.
        aggregate=receiver[:,None,:,:]*((attention@sender[:,None,:,:])
            +torch.einsum('bhij,bijr->bhir',attention,edge_message))
        messages=torch.stack([projection(aggregate[:,a]) for a,projection in enumerate(self.relation_value)],1)
        attended=((attention@v)+messages).transpose(1,2).reshape(b,n,w)
        local=self.local(torch.cat((adjacency@z,group@z),-1))+self.mib_local(mib@z)
        h=h+self.residual_dropout(self.output(attended)+local)
        ff=self.ff[2](self.ff_dropout(self.ff[1](self.ff[0](self.norm2(h)))))
        return (h+self.residual_dropout(ff))*valid[:,:n,None]

class RoleInitial(nn.Module):
    def __init__(self,width=128,layers=3,rectangle=True,ffn_dropout=.1,residual_dropout=.05,initial_layers=3):
        super().__init__()
        if layers!=3 or initial_layers!=3:raise ValueError('E160 requires exactly three attention/FFN blocks')
        self.input=nn.Sequential(nn.Linear(17,width),nn.SiLU(),nn.LayerNorm(width))
        self.role_input=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.LayerNorm(width))
        self.port_input=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.LayerNorm(width))
        self.initial_blocks=nn.ModuleList([RoleBlock(width,heads=4,ffn_dropout=ffn_dropout,residual_dropout=residual_dropout) for _ in range(3)])
        self.head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))

    def encode(self,g):
        if set(g)!=set(PUBLIC):raise ValueError('Only registered public role inputs accepted')
        n=g['valid'].shape[1]
        modules=(self.input(g['features'])+self.role_input(g['role_features']))*g['valid'][...,None]
        ports=self.port_input(g['port_features'])*g['port_valid'][...,None]
        h=modules;valid=torch.cat((g['valid'],g['port_valid']),1)
        edges=g['role_edges'];hops=g['role_hops'];t=edges.shape[2];ids=torch.arange(t,device=edges.device)
        eye=(ids[:n,None]==ids[None,:]).to(edges.dtype)[None].expand(edges.shape[0],-1,-1)
        mib=torch.nn.functional.pad(g['mib'],(0,t-n))
        pair=torch.stack((edges.log1p(),(edges>0).to(edges.dtype),(g['role_group']>0).to(edges.dtype),
            hops.clamp_min(0).to(edges.dtype).asinh(),(hops>=0).to(edges.dtype),eye,(mib>0).to(edges.dtype)),-1)
        adjacency=edges/edges.sum(-1,keepdim=True).clamp_min(1e-6)
        for block in self.initial_blocks:h=block(h,ports,valid,pair,adjacency,g['role_group'],mib)
        return h[:,:n]

    def forward(self,g):
        state=self.head(self.encode(g))*g['valid'][...,None]
        return state[None]

def GeometryModel(architecture='e160',**configuration):
    if architecture!='e160':raise ValueError(architecture)
    return RoleInitial(**configuration)
