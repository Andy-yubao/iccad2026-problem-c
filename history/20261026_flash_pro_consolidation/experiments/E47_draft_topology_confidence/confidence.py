"""Public geometry guidance; optional witness annotations are diagnostics only.

The solver receives only the original draft, candidate graph and confidence
arrays. No witness positions, reference scores or baseline denominators survive
in Guidance. Relative terms are translation invariant and always soft.
"""
from dataclasses import dataclass,field
import hashlib
import numpy as np
from scipy.sparse import coo_matrix,csc_matrix


def candidate_pairs(draft,p):
    center=draft[:,:2]+draft[:,2:]/2;span=np.maximum(np.ptp(center,axis=0),1e-8)
    distance=np.sum(((center[:,None]-center[None,:])/span)**2,2);np.fill_diagonal(distance,np.inf)
    landmarks=np.unique(np.r_[center.argmin(0),center.argmax(0)]);pairs=set()
    for i in range(p.n):
        for j in np.r_[np.argsort(distance[i],kind='stable')[:min(2,p.n-1)],landmarks]:
            if i!=j:pairs.add((min(i,int(j)),max(i,int(j))))
    return np.asarray(sorted(pairs),int).reshape(-1,2)


def confidence_arrays(draft,p,pairs,kind,witness=None,seed=0):
    c=draft[:,:2]+draft[:,2:]/2;i,j=pairs.T;extent=(draft[i,2:]+draft[j,2:])/2;delta=c[i]-c[j]
    span=np.maximum((draft[:,:2]+draft[:,2:]).max(0)-draft[:,:2].min(0),1e-8)
    margin=(abs(delta)-extent)/np.maximum(extent,1e-8);axis=margin.argmax(1)
    if kind=='uniform':return np.full(len(i),.5),np.full((p.n,p.n),.5)
    if kind=='geometric':
        metric=1/(1+np.exp(-np.clip(4*margin[np.arange(len(i)),axis],-20,20)))
        lo,hi=draft[:,:2],draft[:,:2]+draft[:,2:]
        gap=np.maximum(lo[:,None]-hi[None],lo[None]-hi[:,None]);unit=np.minimum(np.sqrt(p.areas)[:,None],np.sqrt(p.areas)[None,:])
        cost=np.maximum(gap,0).sum(2)/np.maximum(unit,1e-8)
        contact=np.clip(np.exp(-2*cost),.02,.98)
    else:
        if witness is None:raise ValueError('Witness required only for explicit auxiliary diagnostic')
        wc=witness[:,:2]+witness[:,2:]/2;error=np.linalg.norm(((wc[i]-wc[j])-delta)/span,axis=1)
        sign=np.sign(delta[np.arange(len(i)),axis]);signed=(wc[i]-wc[j])[np.arange(len(i)),axis]*sign
        compatible=signed-(witness[i,2:]+witness[j,2:])[np.arange(len(i)),axis]/2>=-1e-7
        metric=np.clip((1/(1+(error/.15)**2))*np.where(compatible,1.,.05),.02,.98)
        ov=np.minimum((witness[:,:2]+witness[:,2:])[:,None],(witness[:,:2]+witness[:,2:])[None])-np.maximum(witness[:,None,:2],witness[None,:,:2])
        touch=((abs(ov[:,:,0])<=1e-7)&(ov[:,:,1]>1e-7))|((abs(ov[:,:,1])<=1e-7)&(ov[:,:,0]>1e-7))
        contact=np.where(touch,.98,.02)
        if kind=='shuffle':
            rng=np.random.default_rng(seed);metric=metric[rng.permutation(len(metric))]
            for g in np.unique(p.constraints[:,3]):
                if g<=0:continue
                ids=np.flatnonzero(p.constraints[:,3]==g);a,b=np.tril_indices(len(ids),-1);a,b=ids[a],ids[b]
                values=contact[a,b].copy();values=values[rng.permutation(len(values))];contact[a,b]=values;contact[b,a]=values
    return metric,contact


@dataclass(frozen=True)
class Guidance:
    original:np.ndarray
    pairs:np.ndarray
    delta:np.ndarray
    span:np.ndarray
    metric_confidence:np.ndarray
    contact_confidence:np.ndarray
    metric_enabled:bool=True
    contact_enabled:bool=True
    strength:float=.5
    priority:bool=False
    commit:bool=False
    edge_matrix:object=None
    disabled:set=field(default_factory=set)
    committed:set=field(default_factory=set)
    events:list=field(default_factory=list)
    static_edges:object=None
    global_edges:object=None
    topology_confidence:object=None
    global_mode:str=''
    active_mask:object=None

    @classmethod
    def build(cls,draft,p,kind,witness=None,seed=0):
        if kind in ('draft_static','oracle_full','oracle_hybrid','oracle_hybrid_contacts','shuffle_hybrid'):
            from .global_information import build
            return build(cls,draft,p,kind,witness,seed)
        original=np.asarray(draft,float).copy();pairs=candidate_pairs(original,p)
        extended=kind.endswith('_commit') or kind in ('oracle_structured','oracle_priority','oracle_draft_priority')
        basic='geometric' if kind=='geometric_commit' else 'shuffle' if kind in ('shuffle_commit','shuffle_draft_commit') else 'oracle' if extended or kind in ('oracle_metric','oracle_contact') else kind
        metric,contact=confidence_arrays(original,p,pairs,basic,witness,seed)
        z=original/p.scale;center=z[:,:2]+z[:,2:]/2;i,j=pairs.T
        delta=center[i]-center[j];span=np.maximum((z[:,:2]+z[:,2:]).max(0)-z[:,:2].min(0),1e-6)
        from .fast_geometry import topology
        source=witness if kind in ('oracle_structured','oracle_priority','shuffle_commit') else original
        edge_matrix=np.zeros((p.n,p.n,3),int);a,b=np.tril_indices(p.n,-1);full=topology(source,p.constraints,array=True)
        edge_matrix[a,b]=full;edge_matrix[b,a]=full
        for a in (original,pairs,delta,span,metric,contact,edge_matrix):a.flags.writeable=False
        return cls(original,pairs,delta,span,metric,contact,metric_enabled=not extended and kind!='oracle_contact',
            contact_enabled=kind!='oracle_metric',priority=extended,commit=extended and kind not in ('oracle_priority','oracle_draft_priority'),edge_matrix=edge_matrix)

    def confidence(self,i,j):
        values=self.contact_confidence[i,j].copy()
        for k,(a,b) in enumerate(zip(i,j)):
            if (min(int(a),int(b)),max(int(a),int(b))) in self.disabled:values[k]=0.
        return values

    def impose(self,state,p,full):
        if self.global_mode:
            self.committed.clear()
            if self.global_edges is not None:
                self.committed.update((min(int(a),int(b)),max(int(a),int(b))) for a,b,_ in self.global_edges
                    if (min(int(a),int(b)),max(int(a),int(b))) not in self.disabled)
            if self.active_mask is not None:
                if not self.active_mask.any():return full
                out=full.copy();out[self.active_mask]=self.static_edges[self.active_mask]
                from .global_information import acyclic
                if not acyclic(out,p.n):
                    self.events.append({'reason':'predicted_cycle_defer','pair':None});return full
                return out
            return self.static_edges.copy()
        if not self.priority:return full
        from .fast_geometry import contact_tree
        proposed=contact_tree(state,p,full,self);original=full.copy();out=full.copy();self.committed.clear()
        for a,b,axis in proposed:
            pair=(min(a,b),max(a,b))
            if pair in self.disabled or self.contact_confidence[a,b]<.9:continue
            edge=self.edge_matrix[a,b];u,v,axis=map(int,edge)
            if p.preplaced[u] and p.preplaced[v]:
                tangent=1-axis;gap=p.specs[v,axis]-p.specs[u,axis]-p.specs[u,axis+2]
                overlap=min(p.specs[u,tangent]+p.specs[u,tangent+2],p.specs[v,tangent]+p.specs[v,tangent+2])-max(p.specs[u,tangent],p.specs[v,tangent])
                if abs(gap)>1e-7 or overlap<=0:
                    self.disabled.add(pair);self.events.append({'reason':'immutable_contact','pair':pair});continue
            out[pair[1]*(pair[1]-1)//2+pair[0]]=edge;self.committed.add(pair)
        for _ in range(len(self.committed)+1):
            bad=set()
            for axis in (0,1):
                part=out[out[:,2]==axis];adj=[[] for _ in range(p.n)];degree=np.zeros(p.n,int)
                for a,b,_ in part:adj[a].append(b);degree[b]+=1
                todo=list(np.flatnonzero(degree==0))
                while todo:
                    a=todo.pop()
                    for b in adj[a]:
                        degree[b]-=1
                        if degree[b]==0:todo.append(b)
                nodes=set(np.flatnonzero(degree>0));bad.update(pair for pair in self.committed if self.edge_matrix[pair][2]==axis and pair[0] in nodes and pair[1] in nodes)
            if not bad:break
            pair=min(bad,key=lambda x:(self.contact_confidence[x],x));self.disabled.add(pair);self.committed.remove(pair)
            self.events.append({'reason':'axis_cycle','pair':pair})
            at=pair[1]*(pair[1]-1)//2+pair[0];out[at]=original[at]
        return out

    def hard_contacts(self,tree):
        if not self.commit:return np.zeros((0,3),int)
        if self.global_edges is not None:tree=self.global_edges
        return np.asarray([edge for edge in tree if (min(int(edge[0]),int(edge[1])),max(int(edge[0]),int(edge[1]))) in self.committed],int).reshape(-1,3)

    def release_one(self):
        if not self.committed:return False
        pair=min(self.committed,key=lambda x:(self.contact_confidence[x],x));self.disabled.add(pair);self.committed.remove(pair)
        self.events.append({'reason':'solver_failure','pair':pair});return True

    def quadratic(self,xy,nv,dims=None,fixed_wh=None):
        if not self.metric_enabled or not len(self.pairs):return csc_matrix((nv,nv)),np.zeros(nv)
        i,j=self.pairs.T;norm=np.maximum(np.sum((self.delta/self.span)**2,1),.25)
        weight=self.strength*self.metric_confidence[:,None]/(len(i)*norm[:,None]*self.span[None,:]**2)
        q=np.zeros(nv);rr=[];cc=[];vv=[]
        for axis in (0,1):
            cols=np.column_stack((xy[i,axis],xy[j,axis]));factors=[1.,-1.];target=self.delta[:,axis].copy()
            if dims is None:target-=.5*(fixed_wh[i,axis]-fixed_wh[j,axis])
            else:cols=np.column_stack((cols,dims[i,axis],dims[j,axis]));factors+=[.5,-.5]
            for k,a in enumerate(factors):
                np.add.at(q,cols[:,k],-2*weight[:,axis]*target*a)
                for h,b in enumerate(factors):rr.append(cols[:,k]);cc.append(cols[:,h]);vv.append(2*weight[:,axis]*a*b)
        return coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(nv,nv)).tocsc(),q

    def diagnostics(self):
        return {'original_sha256':hashlib.sha256(self.original.tobytes()).hexdigest(),'pairs':len(self.pairs),
            'metric_confidence_mean':float(self.metric_confidence.mean()),'metric_enabled':self.metric_enabled,'contact_enabled':self.contact_enabled,
            'strength':self.strength,'priority':self.priority,'commit':self.commit,
            'committed_final':sorted(self.committed),'disabled_final':sorted(self.disabled),'retractions':self.events,
            'global_mode':self.global_mode,'global_contacts':0 if self.global_edges is None else len(self.global_edges),
            'active_predicted_relations':None if self.active_mask is None else int(self.active_mask.sum()),
            'original_topology_compatible_fraction':None if self.topology_confidence is None else float(np.mean(self.topology_confidence>=.9)),
            'topology_changes_from_original':None if self.static_edges is None else int(np.any(self.static_edges!=self.edge_matrix[np.tril_indices(len(self.original),-1)],axis=1).sum()),
            'witness_coordinates_in_solver_context':False}
