"""Predict repair relations without changing frozen C1 physical geometry."""
import numpy as np
import torch
from torch import nn
from .student import DeploymentModel as StudentDeployment
from .nn85.corrector import Corrector
from .nn85.base_model import decode

class RelationHead(Corrector):
    def __init__(self,width=192,message_width=64,dropout=.05):
        super().__init__(width,message_width,dropout)
        self.output[-1]=nn.Linear(width,2)
        self.axis=nn.Linear(message_width,2)
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.axis.weight);nn.init.zeros_(self.axis.bias)
    def forward(self,state,g):
        p=decode(state,g);local=g['area'].clamp_min(1e-12).sqrt();center=p[...,:2]+p[...,2:]/2
        shape=(p[...,2:]/local[...,None]).clamp_min(1e-6).log().clamp(-5,5)
        h=self.node(torch.cat((g['features'],state,shape),-1))*g['valid'][...,None]
        size=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
        delta=(center[:,:,None]-center[:,None,:])/size[...,None]
        gap=delta.abs()-(p[:,:,None,2:]+p[:,None,:,2:])/(2*size[...,None])
        n=state.shape[1];flags=g['constraints'][...,:2].to(state.dtype)
        edge=torch.cat((delta.clamp(-8,8),gap.clamp(-8,8),(-gap).clamp(0,8),shape[:,:,None].expand(-1,-1,n,-1),shape[:,None,:].expand(-1,n,-1,-1),torch.stack((g['adjacency'],(g['mib']>0).to(state.dtype),(g['group']>0).to(state.dtype)),-1),flags[:,:,None].expand(-1,-1,n,-1),flags[:,None,:].expand(-1,n,-1,-1)),-1)
        value=torch.nn.functional.silu(self.receiver(h)[:,:,None]+self.sender(h)[:,None,:]+self.edge(edge))
        valid=g['valid'][:,:,None]&g['valid'][:,None,:]&~torch.eye(n,device=state.device,dtype=torch.bool)[None]
        near=(gap.clamp_min(0).sum(-1)<=.5)|(g['adjacency']>0)|(g['group']>0)|(g['mib']>0)
        scores=self.score(value).squeeze(-1).masked_fill(~(valid&near),-1e4)
        attention=scores.softmax(-1)*(valid&near);attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        message=(value*attention[...,None]).sum(2)
        rank=center+.5*local[...,None]*self.output(torch.cat((h,message),-1))*g['mask'][...,:2]
        rank=rank*g['valid'][...,None]
        extent=(p[:,:,None,2:]+p[:,None,:,2:])/2
        native=(center[:,:,None]-center[:,None,:]).abs()/extent.clamp_min(1e-8)-1
        prior=native/(1+native.abs())
        residual=self.axis(value)
        free_pair=g['mask'][...,0].bool()[:,:,None]|g['mask'][...,0].bool()[:,None,:]
        logits=prior+(residual+residual.transpose(1,2))/2*free_pair[...,None]
        return rank,logits

class RelationRefiner(nn.Module):
    def __init__(self,draft,width,message_width,dropout):
        super().__init__();self.draft=draft;self.head=RelationHead(width,message_width,dropout)
    def train(self,mode=True):
        super().train(mode);self.draft.eval();return self
    def forward(self,base,g):
        with torch.no_grad():state=self.draft(base,g)
        rank,logits=self.head(state,g)
        return torch.cat((state,rank,logits.flatten(2)),-1)

class DeploymentModel(StudentDeployment):
    def __init__(self,width=192,layers=12,rectangle=True,ffn_dropout=.05,corrector_width=192,message_width=64,max_correction=.15,cooperative=True):
        super().__init__(width,layers,rectangle,ffn_dropout,corrector_width,message_width,max_correction,cooperative)
        self.refiner=RelationRefiner(self.refiner,corrector_width,message_width,ffn_dropout)

def relations(packed,n):
    """One axis per unordered pair; scalar rank+ID tie-break yields per-axis DAGs."""
    p=np.asarray(packed);slots=(p.shape[-1]-5)//2;rank=p[:n,3:5]
    logits=p[:n,5:].reshape(n,slots,2)[:,:n];i,j=np.tril_indices(n,-1)
    axis=np.argmax(logits[i,j],axis=1)
    sign=rank[i,axis]>=rank[j,axis]
    return np.column_stack((np.where(sign,j,i),np.where(sign,i,j),axis))
