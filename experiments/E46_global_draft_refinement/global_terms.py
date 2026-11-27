"""Original-draft global, translation-invariant terms; public geometry only."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


@dataclass(frozen=True)
class GlobalContext:
    mode: str
    span: np.ndarray
    centers: np.ndarray
    pairs: np.ndarray
    target_wh: np.ndarray

    @classmethod
    def build(cls, draft, p, mode):
        z=np.asarray(draft,float)/p.scale
        span=np.maximum((z[:,:2]+z[:,2:]).max(0)-z[:,:2].min(0),1e-6)
        centers=z[:,:2]+z[:,2:]/2
        pairs=set()
        # Long-range landmarks constrain global arrangement; nearest pairs
        # supplement local structure without assigning absolute positions.
        landmarks=np.unique(np.r_[centers.argmin(0),centers.argmax(0)])
        distance=((centers[:,None]-centers[None,:])/span)**2
        distance=distance.sum(2);np.fill_diagonal(distance,np.inf)
        for i in range(p.n):
            for j in np.r_[np.argsort(distance[i],kind='stable')[:min(2,p.n-1)],landmarks]:
                if i!=j:pairs.add((min(i,int(j)),max(i,int(j))))
        pairs=np.asarray(sorted(pairs),int).reshape(-1,2)
        from .backend.frozen.packing import _dimensions
        w,h=_dimensions(p,draft[:,2]/draft[:,3],True)
        target_wh=np.column_stack((w,h))/p.scale
        for v in (span,centers,pairs,target_wh):v.flags.writeable=False
        return cls(mode,span,centers,pairs,target_wh)

    def envelope(self,ineq,costs,lo,hi,slack):
        # Hinge penalty only for >5% excess span. Translation and shrinkage
        # remain free, and slack prevents an infeasible raw draft envelope.
        if self.mode!='envelope':return
        ineq.add(np.column_stack((hi,lo,slack)),[1,-1,-1],1.05*self.span)
        costs[slack]=.75/self.span

    def quadratic(self,xy,nv,dims=None,fixed_wh=None):
        if self.mode=='shape_memory' and dims is not None:
            weight=.02/(len(dims)*np.maximum(self.target_wh,1e-9)**2)
            ids=dims.ravel();q=np.zeros(nv);q[ids]=-2*(weight*self.target_wh).ravel()
            return coo_matrix(((2*weight).ravel(),(ids,ids)),shape=(nv,nv)).tocsc(),q
        if self.mode!='relative' or not len(self.pairs):
            return csc_matrix((nv,nv)),np.zeros(nv)
        i,j=self.pairs.T
        delta=self.centers[i]-self.centers[j]
        norm=np.maximum(np.sum((delta/self.span)**2,1),.25)
        weight=.5/(len(i)*norm[:,None]*self.span[None,:]**2)
        q=np.zeros(nv);rr=[];cc=[];vv=[]
        for a in (0,1):
            cols=np.column_stack((xy[i,a],xy[j,a]))
            factors=np.array([1.,-1.]);target=delta[:,a].copy()
            if dims is not None:
                cols=np.column_stack((cols,dims[i,a],dims[j,a]));factors=np.r_[factors,.5,-.5]
            else:target-=.5*(fixed_wh[i,a]-fixed_wh[j,a])
            for k,coef in enumerate(factors):
                np.add.at(q,cols[:,k],-2*weight[:,a]*target*coef)
                for h,other in enumerate(factors):
                    rr.append(cols[:,k]);cc.append(cols[:,h]);vv.append(2*weight[:,a]*coef*other)
        return coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(nv,nv)).tocsc(),q

    def shape_bounds(self,ineq,dims,free):
        if self.mode=='shape_trust':
            ineq.add(dims[free].ravel(),-1,-.75*self.target_wh[free].ravel())
            ineq.add(dims[free].ravel(),1,(4/3)*self.target_wh[free].ravel())

    def energy(self,centers):
        if not len(self.pairs):return 0.
        i,j=self.pairs.T;target=self.centers[i]-self.centers[j]
        norm=np.maximum(np.sum((target/self.span)**2,1),.25)
        error=(centers[i]-centers[j]-target)/self.span
        return float((.5/(len(i)*norm)*np.sum(error**2,1)).sum())
