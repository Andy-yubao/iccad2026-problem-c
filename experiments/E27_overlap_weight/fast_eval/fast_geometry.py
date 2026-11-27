"""Vectorized public-input SOCP assembly; no reference geometry or scores."""
import time
import numpy as np
import clarabel
import heapq
from scipy.sparse import coo_matrix, csc_matrix
from .backend.frozen.feasible import Objective


def topology(pos,c,mode=0,array=False,group_order=False):
    n=len(pos);i,j=np.tril_indices(n,-1);center=pos[:,:2]+pos[:,2:]/2
    delta=center[i]-center[j];extent=(pos[i,2:]+pos[j,2:])/2
    axis=np.argmax((np.abs(delta)-extent)/np.maximum(extent,1e-12),axis=1)
    a,b=c[i,4].astype(int),c[j,4].astype(int)
    if mode:
        axis=np.where((a&b&3)!=0,1,np.where((a&b&12)!=0,0,axis))
    sign=delta[np.arange(len(i)),axis]>=0
    if mode:
        lower=np.where(axis==0,1,8);upper=np.where(axis==0,2,4)
        first=((a&lower)!=0)|((b&upper)!=0);second=((b&lower)!=0)|((a&upper)!=0)
        sign=np.where(first,False,np.where(second,True,sign))
    if group_order:
        macro=center.copy();groups=c[:,3].astype(int)
        for g in np.unique(groups):
            if g>0:
                ids=np.flatnonzero(groups==g);macro[ids]=center[ids].mean(0)
        separate=(groups[i]!=groups[j])|((groups[i]==0)&(groups[j]==0))
        macro_sign=(macro[i]-macro[j])[np.arange(len(i)),axis]>=0
        if mode:
            macro_sign=np.where(first,False,np.where(second,True,macro_sign))
        sign=np.where(separate,macro_sign,sign)
    result=np.column_stack((np.where(sign,j,i),np.where(sign,i,j),axis))
    return result if array else [tuple(row) for row in result.tolist()]


def contact_tree(seed,p,edges):
    """Kruskal only over public-group pairs; avoid O(n^2) frozenset mappings."""
    edges=np.asarray(edges,dtype=int).reshape(-1,3);first=np.zeros((p.n,p.n),dtype=int);axes=np.zeros((p.n,p.n),dtype=int)
    if len(edges):
        a,b,c=edges.T;first[a,b]=a;first[b,a]=a;axes[a,b]=c;axes[b,a]=c
    tree=[]
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==g);ii,jj=np.tril_indices(len(ids),-1)
        i,j=ids[ii],ids[jj];a=first[i,j];b=np.where(a==i,j,i);axis=axes[i,j];t=1-axis
        gap=np.abs(seed[b,axis]-seed[a,axis]-seed[a,axis+2])
        tangent=np.maximum.reduce((np.zeros(len(i)),seed[a,t]-seed[b,t]-seed[b,t+2],seed[b,t]-seed[a,t]-seed[a,t+2]))
        ranking=np.lexsort((axis,b,a,gap+2*tangent));parent=np.arange(p.n)
        def find(i):
            while parent[i]!=i:i=parent[i]
            return i
        for k in ranking:
            x,y=int(a[k]),int(b[k]);rx,ry=find(x),find(y)
            if rx!=ry:parent[rx]=ry;tree.append((x,y,int(axis[k])))
    return tree


def reduce_edges(edges,n):
    """Same positive-size DAG reduction, without full-pair Python tuples."""
    edges=np.asarray(edges,dtype=int).reshape(-1,3);out=[]
    for axis in (0,1):
        part=edges[edges[:,2]==axis];adj=[[] for _ in range(n)];degree=[0]*n
        for i,j in zip(part[:,0].tolist(),part[:,1].tolist()):adj[i].append(j);degree[j]+=1
        queue=[i for i,d in enumerate(degree) if d==0];heapq.heapify(queue);order=[]
        while queue:
            i=heapq.heappop(queue);order.append(i)
            for j in adj[i]:
                degree[j]-=1
                if degree[j]==0:heapq.heappush(queue,j)
        if len(order)!=n:out.extend(tuple(row) for row in part.tolist());continue
        rank=[0]*n
        for k,i in enumerate(order):rank[i]=k
        reach=[0]*n
        for i in reversed(order):
            covered=0
            for j in sorted(adj[i],key=rank.__getitem__):
                if not covered&(1<<j):out.append((i,j,axis))
                covered|=(1<<j)|reach[j]
            reach[i]=covered
    return out


class Rows:
    def __init__(self): self.i=[];self.j=[];self.v=[];self.b=[];self.count=0
    def add(self,columns,values,rhs):
        columns=np.asarray(columns,dtype=int)
        if columns.ndim==1: columns=columns[:,None]
        if len(columns)==0:return
        values=np.broadcast_to(np.asarray(values,dtype=float),columns.shape)
        rhs=np.broadcast_to(np.asarray(rhs,dtype=float),len(columns))
        self.i.append(np.repeat(np.arange(len(columns))+self.count,columns.shape[1]))
        self.j.append(columns.ravel());self.v.append(values.ravel());self.b.append(rhs)
        self.count+=len(columns)


def step(p,seed,edges,*,seconds=.2,boundary=12.,contact=6.,elastic=100000.,
         reweighted=True,conservative=True,full_edges=None,tolerance=1e-10,anchors=None,wire_exact=False):
    start=time.perf_counter();n=p.n;s=p.scale;z=seed/s;wh=z[:,2:]
    locked=p.fixed|p.preplaced;free=np.flatnonzero(~locked);objective=Objective(p);ns=objective.nsoft
    full_edges=topology(seed,p.constraints,array=True) if full_edges is None else full_edges
    tree=np.asarray(contact_tree(seed,p,full_edges),dtype=int).reshape(-1,3)
    edges=np.asarray(edges,dtype=int).reshape(-1,3);e=len(edges);t=len(tree)
    xy=np.arange(2*n).reshape(n,2);dims=2*n+np.arange(2*n).reshape(n,2)
    lo=4*n+np.arange(2);hi=4*n+2+np.arange(2);move=4*n+4+np.arange(2*n).reshape(n,2)
    requested=np.flatnonzero(p.constraints[:,4]);boundary_ids=6*n+4+np.arange(len(requested))
    elastic_ids=6*n+4+len(requested)+np.arange(e if elastic else 0)
    contact_ids=6*n+4+len(requested)+len(elastic_ids)+np.arange(t)
    nv=6*n+4+len(requested)+len(elastic_ids)+t
    bi,bj=np.nonzero(np.triu(p.adjacency,1));bw=p.adjacency[bi,bj]
    if len(p.p2b): pin,pblock,pweight=p.p2b.T;pin=pin.astype(int);pblock=pblock.astype(int)
    else:pin=pblock=np.zeros(0,dtype=int);pweight=np.zeros(0)
    if wire_exact:
        wb=nv+np.arange(2*len(bi)).reshape(-1,2);nv+=2*len(bi)
        wp=nv+np.arange(2*len(pin)).reshape(-1,2);nv+=2*len(pin)
    costs=np.zeros(nv)
    box_weight=np.ones(2) if anchors is None else .5*((z[:,:2]+wh).max(0)-z[:,:2].min(0))[::-1]/anchors[0]
    costs[lo]=-box_weight;costs[hi]=box_weight;costs[move]=.005/n
    if elastic:costs[elastic_ids]=elastic/n
    eq=Rows();ineq=Rows()
    eq.add(dims[locked].ravel(),1,wh[locked].ravel())
    eq.add(xy[p.preplaced].ravel(),1,z[p.preplaced,:2].ravel())
    ineq.add(np.arange(nv),-1,0.)
    ineq.add(dims[free].ravel(),-1,-.4*wh[free].ravel())
    ineq.add(dims[free].ravel(),1,2.5*wh[free].ravel())
    if conservative:
        ineq.add(dims[free],wh[free,::-1],2*np.sqrt(wh[free].prod(1)*(p.areas[free]/s**2)*1.009999))
    # Compatible MIB classes have one common dimension pair.
    parent=np.arange(n)
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for group in np.unique(p.constraints[:,2]):
        if group<=0:continue
        ids=np.flatnonzero(p.constraints[:,2]==group)
        for at,i in enumerate(ids):
            same=ids[:at][np.max(np.abs(wh[ids[:at]]-wh[i]),axis=1)<1e-10]
            if len(same):
                j=same[0];eq.add(np.stack((dims[i],dims[j]),axis=1),[1,-1],0);parent[root(i)]=root(j)
    all_i=np.repeat(np.arange(n),2);axis=np.tile(np.arange(2),n)
    ineq.add(np.column_stack((lo[axis],xy.ravel())),[1,-1],0)
    ineq.add(np.column_stack((xy.ravel(),dims.ravel(),hi[axis])),[1,1,-1],0)
    ineq.add(np.column_stack((xy.ravel(),move.ravel())),[1,-1],z[:,:2].ravel())
    ineq.add(np.column_stack((xy.ravel(),move.ravel())),[-1,-1],-z[:,:2].ravel())
    if len(requested):
        bblo=z[:,:2].min(0);bbhi=(z[:,:2]+wh).max(0)
        distances=np.column_stack((z[:,0]-bblo[0],bbhi[0]-z[:,0]-wh[:,0],bbhi[1]-z[:,1]-wh[:,1],z[:,1]-bblo[1]))
        bits=np.array([1,2,4,8]);mask=(p.constraints[requested,4,None].astype(int)&bits)!=0
        prior=np.where(mask,distances[requested],-np.inf).max(1)
        costs[boundary_ids]=2/ns/(np.maximum(prior,0)+.02*np.sqrt(wh[requested].prod(1))) if reweighted else boundary/ns
        for column,(a,upper) in enumerate([(0,False),(0,True),(1,True),(1,False)]):
            use=mask[:,column];ids=requested[use];slack=boundary_ids[use]
            if upper:ineq.add(np.column_stack((np.full(len(ids),hi[a]),xy[ids,a],dims[ids,a],slack)),[1,-1,-1,-1],0)
            else:ineq.add(np.column_stack((xy[ids,a],np.full(len(ids),lo[a]),slack)),[1,-1,-1],0)
    edge_row_start=ineq.count
    if e:
        i,j,a=edges.T
        columns=np.column_stack((xy[i,a],dims[i,a],xy[j,a]))
        if elastic:ineq.add(np.column_stack((columns,elastic_ids)),[1,1,-1,-1],0)
        else:ineq.add(columns,[1,1,-1],0)
    if t:
        i,j,a=tree.T;b=1-a
        gap=np.maximum.reduce((np.zeros(t),z[j,a]-z[i,a]-wh[i,a],z[i,b]-z[j,b]-wh[j,b],z[j,b]-z[i,b]-wh[i,b]))
        costs[contact_ids]=2/ns/(gap+.02*np.minimum(np.sqrt(wh[i].prod(1)),np.sqrt(wh[j].prod(1)))) if reweighted else contact/ns
        shared=.02*np.minimum(wh[i,b],wh[j,b])
        ineq.add(np.column_stack((xy[j,a],xy[i,a],dims[i,a],contact_ids)),[1,-1,-1,-1],0)
        ineq.add(np.column_stack((xy[i,b],xy[j,b],dims[j,b],contact_ids)),[1,-1,-1,-1],-shared)
        ineq.add(np.column_stack((xy[j,b],xy[i,b],dims[i,b],contact_ids)),[1,-1,-1,-1],-shared)
    # IRLS diagonal majorization, aggregated per block before sparse assembly.
    centers=z[:,:2]+wh/2;alpha=np.zeros((n,2));beta=np.zeros((n,2))
    norm=max(p.total_wire_weight if anchors is None else 2*anchors[1],1e-12)
    i,j=np.nonzero(np.triu(p.adjacency,1));w=p.adjacency[i,j]/norm
    if len(i):
        prior=centers[i]-centers[j];local=w[:,None]/np.sqrt(prior**2+.02**2);mid=(centers[i]+centers[j])/2
        np.add.at(alpha,i,local);np.add.at(alpha,j,local);np.add.at(beta,i,local*mid);np.add.at(beta,j,local*mid)
    if len(p.p2b):
        pin,i,w=p.p2b.T;pin=pin.astype(int);i=i.astype(int);target=p.pins[pin,:2]/s
        local=.5*(w/norm)[:,None]/np.sqrt((centers[i]-target)**2+.02**2)
        np.add.at(alpha,i,local);np.add.at(beta,i,local*target)
    if wire_exact:
        costs[wb]=bw[:,None]/norm;costs[wp]=pweight[:,None]/norm
        for a in (0,1):
            columns=np.column_stack((xy[bi,a],dims[bi,a],xy[bj,a],dims[bj,a],wb[:,a]))
            ineq.add(columns,[1,.5,-1,-.5,-1],0)
            ineq.add(columns,[-1,-.5,1,.5,-1],0)
            columns=np.column_stack((xy[pblock,a],dims[pblock,a],wp[:,a]))
            ineq.add(columns,[1,.5,-1],p.pins[pin,a]/s)
            ineq.add(columns,[-1,-.5,-1],-p.pins[pin,a]/s)
        hessian=csc_matrix((nv,nv))
    else:
        costs[xy]-=2*beta;costs[dims]-=beta
        pi=np.r_[xy.ravel(),dims.ravel(),xy.ravel()];pj=np.r_[xy.ravel(),dims.ravel(),dims.ravel()]
        pv=np.r_[(2*alpha).ravel(),(.5*alpha).ravel(),alpha.ravel()]
        hessian=coo_matrix((pv,(pi,pj)),shape=(nv,nv)).tocsc()
    ii=[];jj=[];vv=[];rhs=[];offset=0
    for block in (eq,ineq):
        if block.count:
            ii.append(np.concatenate(block.i)+offset);jj.extend(block.j);vv.extend(block.v);rhs.extend(block.b)
        offset+=block.count
    cones=[clarabel.ZeroConeT(eq.count),clarabel.NonnegativeConeT(ineq.count)]
    if len(free):
        rr=offset+3*np.arange(len(free));ii.append(np.column_stack((rr,rr,rr+1,rr+1)).ravel())
        jj.append(np.column_stack((dims[free,0],dims[free,1],dims[free,0],dims[free,1])).ravel())
        vv.append(np.tile([-1,-1,-1,1],len(free)))
        cone_rhs=np.column_stack((np.zeros((len(free),2)),2*np.sqrt(p.areas[free]/s**2*.990001))).ravel();rhs.append(cone_rhs)
        cones.extend(clarabel.SecondOrderConeT(3) for _ in free);offset+=3*len(free)
    matrix=coo_matrix((np.concatenate(vv),(np.concatenate(ii),np.concatenate(jj))),shape=(offset,nv)).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=100
    settings.tol_feas=tolerance;settings.tol_gap_abs=tolerance;settings.tol_gap_rel=tolerance
    settings.time_limit=max(.005,seconds-(time.perf_counter()-start))
    assembled=time.perf_counter();solver=clarabel.DefaultSolver(hessian,costs,matrix,np.concatenate(rhs),cones,settings)
    initialized=time.perf_counter();result=solver.solve()
    stats={'status':str(result.status),'native_seconds':result.solve_time,'seconds':time.perf_counter()-start,
        'assembly_seconds':assembled-start,'setup_seconds':initialized-assembled,
        'iterations':result.iterations,'variables':nv,'constraints':offset}
    if str(result.status)=='PrimalInfeasible' and not elastic and e:
        ray=np.asarray(result.z)[eq.count+edge_row_start:eq.count+edge_row_start+e]
        ranked=np.argsort(-np.abs(ray))[:8]
        stats['conflicting_edges']=[[int(edges[k,0]),int(edges[k,1]),int(edges[k,2]),float(abs(ray[k]))] for k in ranked]
    if str(result.status) not in ('Solved','AlmostSolved','MaxTime') or result.r_prim>2e-6:return None,stats
    values=np.asarray(result.x);out=np.column_stack((values[xy],values[dims]))*s
    out[locked,2:]=p.specs[locked,2:];out[p.preplaced]=p.specs[p.preplaced]
    classes={}
    for i in range(n):classes.setdefault(root(i),[]).append(i)
    for members in classes.values():
        ids=np.asarray(members);f=ids[~locked[ids]]
        if not len(f):continue
        actual=out[f,2:].prod(1);lower=float(np.max(p.areas[f]*.9900001/actual));upper=float(np.min(p.areas[f]*1.009999/actual))
        factor=1. if locked[ids].any() or lower>upper else float(np.clip(1.,lower,upper))
        out[f,2:]*=np.sqrt(factor)
    return out,stats
