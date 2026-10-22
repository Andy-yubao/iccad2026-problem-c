"""Public latent/state statistics and a small solver-selection network."""
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .cases import graph
from .model import decode

@torch.no_grad()
def prefix(model,inputs):
    started=time.perf_counter();g=graph(inputs);state,h=model.forward_with_hidden(g)
    positions=(decode(state,g)[0]*g['scale'][0]).numpy().astype(float)
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1);positions[locked,2:]=s[locked,2:];positions[c[:,1]!=0]=s[c[:,1]!=0]
    elapsed=time.perf_counter()-started
    return positions,h[0].numpy().astype(float),elapsed

def statistics(inputs,draft,hidden):
    n,areas,b2b,p2b,pins,c,s=inputs;n=int(n);areas=np.asarray(areas);c=np.asarray(c);scale=np.sqrt(areas.sum());x=draft/scale
    center=x[:,:2]+x[:,2:]/2;extent=(x[:,None,2:]+x[None,:,2:])/2;delta=np.abs(center[:,None]-center[None,:]);gap=delta-extent
    local=(np.sqrt(areas[:,None])+np.sqrt(areas[None,:]))/(2*scale);penetration=np.maximum(0,-gap.max(-1))/np.maximum(local,1e-9);np.fill_diagonal(penetration,0)
    overlap=((gap<0).all(-1));np.fill_diagonal(overlap,False)
    lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0)
    sides=np.column_stack((x[:,0]-lo[0],hi[0]-x[:,0]-x[:,2],hi[1]-x[:,1]-x[:,3],x[:,1]-lo[1]))/np.sqrt(areas/areas.sum())[:,None]
    def stats(a):
        a=np.asarray(a,dtype=float).ravel()
        return [0.,0.,0.,0.] if not len(a) else [float(a.mean()),float(a.std()),float(a.min()),float(a.max())]
    values=[n/120,float(c[:,0].mean()),float(c[:,1].mean()),float(np.prod(hi-lo)),float(overlap.sum()/max(1,n)),float(np.mean(overlap.any(1)))]
    values+=stats(np.log(areas/areas.mean()))+stats(np.log(x[:,2]/x[:,3]))+stats(penetration.sum(1))+stats(overlap.sum(1))
    values+=stats(center[:,0])+stats(center[:,1])+stats(x[:,2])+stats(x[:,3])
    for column,bit in enumerate((1,2,4,8)):
        required=(c[:,4].astype(int)&bit)!=0;values+=[float(required.mean())]+stats(sides[required,column])
    group_sizes=[];group_scatter=[];group_gap=[]
    for group in np.unique(c[:,3]):
        if group<=0:continue
        ids=np.flatnonzero(c[:,3]==group);group_sizes.append(len(ids)/n)
        group_scatter.append(float(np.prod((x[ids,:2]+x[ids,2:]).max(0)-x[ids,:2].min(0))/max(1e-9,np.sum(x[ids,2:].prod(1)))))
        ii,jj=np.tril_indices(len(ids),-1)
        if len(ii):group_gap.extend((np.maximum(gap[ids[ii],ids[jj]],0).sum(-1)/np.maximum(local[ids[ii],ids[jj]],1e-9)).tolist())
    values+=stats(group_sizes)+stats(group_scatter)+stats(group_gap)
    for edges in (b2b,p2b):values+=stats(np.asarray(edges)[:,2]) if len(edges) else [0.]*4
    wire=0.;total_weight=0.;pin_coordinates=np.asarray(pins)
    pin_coordinates=pin_coordinates[:,:2]/scale if pin_coordinates.size else np.zeros((0,2))
    for edges,pin_edges in ((b2b,False),(p2b,True)):
        edges=np.asarray(edges);edges=edges[~(edges==-1).all(1)]
        if not len(edges):continue
        a,b=edges[:,:2].astype(int).T;weight=edges[:,2]
        first=pin_coordinates[a] if pin_edges else center[a]
        wire+=float((np.abs(first-center[b]).sum(1)*weight).sum());total_weight+=float(weight.sum())
    values+=[wire/max(total_weight,1e-9),wire/max(total_weight*float((hi-lo).sum()),1e-9)]
    # The frozen latent was produced in the same forward pass as the draft.
    embedding=np.concatenate((hidden.mean(0),hidden.std(0),hidden.max(0)))
    result=np.r_[values,embedding].astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite public routing features')
    return result

class Gate(nn.Module):
    def __init__(self,dimensions,width=128,actions=4):
        super().__init__();self.dimensions=dimensions;self.width=width;self.actions=actions
        self.body=nn.Sequential(nn.Linear(dimensions,width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.action=nn.Linear(width,actions);self.quality=nn.Linear(width,actions)
        self.register_buffer('mean',torch.zeros(dimensions));self.register_buffer('std',torch.ones(dimensions))
    def forward(self,features):
        h=self.body(((features-self.mean)/self.std.clamp_min(1e-4)).clamp(-10,10));return self.action(h),self.quality(h)

def oracle_actions(q,t,n):
    normalizer=torch.exp((torch.arange(21,121,device=q.device,dtype=q.dtype)-120)/12).mean()
    importance=torch.exp((n.to(q)-120)/12)/normalizer
    utility=importance[:,None]*(q-q[:,:1])+.05*t+.2*(q-1.2).clamp_min(0)
    eligible=q<=q[:,:1]+.05
    return utility.masked_fill(~eligible,1e4).argmin(1)

def objective(model,x,q,t):
    logits,predicted=model(x);target=oracle_actions(q,t,(120*x[:,0]).round())
    logq=q.clamp_min(1e-6).log();regret=10*(logq-logq[:,:1])
    percase=F.cross_entropy(logits,target,reduction='none')+.2*F.smooth_l1_loss(predicted,regret,reduction='none').mean(1)
    return percase,logits,target
