"""Public-state inference, deterministic DAG axes and group spanning trees."""
import time
import numpy as np
import torch
from .cases import graph
from .relations import encode_positions

def choose(state,p,full,scores,*,axis=True,contact=True):
    full=full.copy();n=p.n;i,j=np.tril_indices(n,-1);center=state[:,:2]+state[:,2:]/2
    extent=(state[i,2:]+state[j,2:])/2;delta=center[i]-center[j]
    clear=(np.abs(delta)-extent)/np.maximum(extent,1e-12)
    z=scores[i,j,:2];prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    predicted=prob.argmax(1);changes=np.zeros(len(full),dtype=bool)
    if axis:
        changes=(prob.max(1)>=.8)&(np.abs(clear[:,0]-clear[:,1])<.5)&(clear.max(1)<.5)&(predicted!=full[:,2])
        a=predicted[changes];ii=i[changes];jj=j[changes];sign=(center[ii]-center[jj])[np.arange(len(a)),a]>=0
        full[changes]=np.column_stack((np.where(sign,jj,ii),np.where(sign,ii,jj),a))
    if not contact:return full,None,{'axis_changes':int(changes.sum()),'contact_tree_edges':0}
    tree=[];groups=p.constraints[:,3].astype(int)
    for group in np.unique(groups):
        if group<=0:continue
        ids=np.flatnonzero(groups==group);ii,jj=np.tril_indices(len(ids),-1);a,b=ids[ii],ids[jj]
        edge_index=np.maximum(a,b)*(np.maximum(a,b)-1)//2+np.minimum(a,b)
        edges=full[edge_index];order=np.lexsort((edges[:,2],edges[:,1],edges[:,0],-scores[a,b,2]));parent=np.arange(n)
        def root(k):
            while parent[k]!=k:k=parent[k]
            return k
        for k in order:
            x,y,ax=map(int,edges[k]);rx,ry=root(x),root(y)
            if rx!=ry:parent[rx]=ry;tree.append((x,y,ax))
    return full,tree,{'axis_changes':int(changes.sum()),'contact_tree_edges':len(tree)}

class LearnedRelations:
    def __init__(self,model,inputs,draft,*,feedback=True,axis=True,contact=True):
        self.model=model;self.g=graph(inputs);self.draft=torch.as_tensor(draft,dtype=torch.float32)[None]/self.g['scale'][:,None,None]
        self.draft=encode_positions(self.draft,self.g);self.feedback=feedback;self.axis=axis;self.contact=contact
        self.scores=None;self.calls=0;self.inferences=0

    @torch.no_grad()
    def __call__(self,state,p,full):
        begin=time.perf_counter();inferred=self.scores is None or (self.feedback and self.calls%3==0)
        if inferred:
            current=encode_positions(torch.as_tensor(state,dtype=torch.float32)[None]/self.g['scale'][:,None,None],self.g)
            self.scores=self.model(self.g,self.draft,current)[0].numpy().astype(float);self.inferences+=1
        self.calls+=1
        edges,tree,info=choose(state,p,full,self.scores,axis=self.axis,contact=self.contact)
        return edges,tree,dict(info,seconds=time.perf_counter()-begin,inferred=inferred,call=self.calls)
