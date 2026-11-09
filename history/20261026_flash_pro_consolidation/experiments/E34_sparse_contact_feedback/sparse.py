"""Same learned modules, only public same-group pair heads are evaluated."""
import time
import numpy as np
import torch
from .relations import RelationPolicy,encode_positions
from .batches import decode
from .cases import graph

class SparsePairPolicy(RelationPolicy):
    def forward(self,g,draft,current,i,j):
        if set(g)-{'features','adjacency','mib','group','mask','known','area','specs','constraints','valid','scale'}:
            raise ValueError('Only public graph fields')
        p=decode(current,g);d=decode(draft,g);valid=g['valid'];local=g['area'].sqrt().clamp_min(1e-6)
        center=p[...,:2]+p[...,2:]/2;dc=d[...,:2]+d[...,2:]/2
        delta=(center[:,:,None]-center[:,None,:]).abs();extent=(p[:,:,None,2:]+p[:,None,:,2:])/2;gap=delta-extent
        pairmask=valid[:,:,None]&valid[:,None,:]&~torch.eye(current.shape[1],device=current.device,dtype=torch.bool)[None]
        penetration=(-gap.amax(-1)).clamp_min(0)*pairmask
        collision=torch.stack((penetration.sum(-1)/local,penetration.amax(-1)/local),-1)
        lo=p[...,:2].masked_fill(~valid[...,None],1e4).amin(1);hi=(p[...,:2]+p[...,2:]).masked_fill(~valid[...,None],-1e4).amax(1)
        sides=torch.stack((p[...,0]-lo[:,None,0],hi[:,None,0]-p[...,0]-p[...,2],hi[:,None,1]-p[...,1]-p[...,3],p[...,1]-lo[:,None,1]),-1)/local[...,None]
        h=self.embed(torch.cat((g['features'],draft,current,collision,sides),-1))*valid[...,None]
        for layer in self.messages:h=(h+layer(torch.cat((h,g['adjacency']@h,g['group']@h,g['mib']@h),-1)))*valid[...,None]
        scale=(local[:,i]+local[:,j])[...,None]/2
        geo=torch.cat((delta[:,i,j]/scale,extent[:,i,j]/scale,gap[:,i,j]/scale,
            (dc[:,i]-dc[:,j]).abs()/scale,(d[:,i,2:]+d[:,j,2:])/2/scale,
            (p[:,i,2:].clamp_min(1e-8).log()-p[:,j,2:].clamp_min(1e-8).log()).abs(),
            ((g['adjacency'][:,i,j]+g['adjacency'][:,j,i])/2)[...,None],
            (g['group'][:,i,j]>0)[...,None],(g['mib'][:,i,j]>0)[...,None]),-1)
        return self.pair(torch.cat((h[:,i]+h[:,j],(h[:,i]-h[:,j]).abs(),geo),-1))

def group_pairs(inputs):
    n=int(inputs[0]);groups=np.asarray(inputs[5])[:,3].astype(int);i,j=np.tril_indices(n,-1);use=(groups[i]>0)&(groups[i]==groups[j])
    return i[use],j[use]

class SparseContacts:
    def __init__(self,model,dense,inputs,draft,*,feedback=True):
        self.model=model;self.dense=dense;self.i,self.j=group_pairs(inputs)
        self.g=graph(inputs) if len(self.i) else None
        self.draft=encode_positions(torch.as_tensor(draft,dtype=torch.float32)[None]/self.g['scale'][:,None,None],self.g) if len(self.i) else None
        self.ii=torch.as_tensor(self.i,dtype=torch.long);self.jj=torch.as_tensor(self.j,dtype=torch.long)
        groups=np.asarray(inputs[5])[:,3].astype(int);self.group_indices=[np.flatnonzero(groups[self.i]==group) for group in np.unique(groups[self.i])]
        self.feedback=feedback;self.scores=None;self.calls=0;self.inferences=0;self.dense_fallbacks=0

    @torch.no_grad()
    def __call__(self,state,p,full):
        begin=time.perf_counter();inferred=self.scores is None or (self.feedback and self.calls%3==0);fallback=False
        if inferred:
            self.inferences+=int(len(self.i)>0)
            if len(self.i):
                current=encode_positions(torch.as_tensor(state,dtype=torch.float32)[None]/self.g['scale'][:,None,None],self.g)
                values=self.model(self.g,self.draft,current,self.ii,self.jj)[0,:,2].numpy().astype(float)
                fallback=any(len(ids)>1 and np.any(np.diff(np.sort(values[ids]))<1e-4) for ids in self.group_indices)
                if fallback:values=self.dense(self.g,self.draft,current)[0,self.ii,self.jj,2].numpy().astype(float);self.dense_fallbacks+=1
                self.scores=values
            else:self.scores=np.zeros(0)
        self.calls+=1;tree=[]
        edge_indices=np.maximum(self.i,self.j)*(np.maximum(self.i,self.j)-1)//2+np.minimum(self.i,self.j)
        for ids in self.group_indices:
            edges=full[edge_indices[ids]];order=np.lexsort((edges[:,2],edges[:,1],edges[:,0],-self.scores[ids]));parent=np.arange(p.n)
            def root(k):
                while parent[k]!=k:k=parent[k]
                return k
            for k in order:
                x,y,axis=map(int,edges[k]);a,b=root(x),root(y)
                if a!=b:parent[a]=b;tree.append((x,y,axis))
        return full,tree,{'seconds':time.perf_counter()-begin,'inferred':inferred and len(self.i)>0,'group_pairs':len(self.i),'dense_fallback':fallback,'call':self.calls,'axis_changes':0,'contact_tree_edges':len(tree)}
