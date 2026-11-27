"""Exact soft targets via coordinated translations; public certificate only."""
import math,time
import numpy as np
import clarabel
from scipy.sparse import coo_matrix,diags
from .fast_geometry import Rows,topology,reduce_edges
from .backend.frozen.feasible import Objective
from .backend.frozen.solver import hard_valid
from .solver import raw_measures
from .canonical import canonicalize

BITS=[(1,0,False),(2,0,True),(4,1,True),(8,1,False)]

def components(x,p):
    parent=np.arange(p.n)
    def root(i):
        while parent[i]!=i:i=int(parent[i])
        return i
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==g)
        lo=x[ids,:2];hi=lo+x[ids,2:]
        ov=np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None])
        touch=((ov[:,:,0]>=0)&(ov[:,:,1]>0))|((ov[:,:,1]>=0)&(ov[:,:,0]>0))
        for a,b in zip(*np.nonzero(np.triu(touch,1))):parent[root(ids[b])]=root(ids[a])
    labels=np.array([root(i) for i in range(p.n)])
    return labels,[np.flatnonzero(labels==g) for g in np.unique(labels)]

def touches(x):
    lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0)
    d=np.column_stack((x[:,0]-lo[0],hi[0]-x[:,0]-x[:,2],hi[1]-x[:,1]-x[:,3],x[:,1]-lo[1]))
    return np.abs(d)<1e-6,lo,hi

def contact_forest(x,p):
    parent=np.arange(p.n);edges=[]
    def root(i):
        while parent[i]!=i:i=int(parent[i])
        return i
    candidates=[]
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==g)
        for at,i in enumerate(ids):
            for j in ids[:at]:
                overlap=np.minimum(x[i,:2]+x[i,2:],x[j,:2]+x[j,2:])-np.maximum(x[i,:2],x[j,:2])
                if not ((overlap[0]>=0 and overlap[1]>0) or (overlap[1]>=0 and overlap[0]>0)):continue
                choices=[]
                for a in [0,1]:
                    if overlap[1-a]<=0:continue
                    for first,second in [(i,j),(j,i)]:
                        gap=abs(x[second,a]-x[first,a]-x[first,a+2])
                        choices.append((gap,first,second,a))
                gap,first,second,a=min(choices)
                candidates.append((-overlap[1-a],int(first),int(second),a))
    for _,i,j,a in sorted(candidates):
        ri,rj=root(i),root(j)
        if ri!=rj:parent[rj]=ri;edges.append((i,j,a))
    return edges

def proposals(x,p,labels,groups,flexible=False):
    touch,lo,hi=touches(x);out=[];scale=p.scale
    for i in np.flatnonzero(p.constraints[:,4]):
        code=int(p.constraints[i,4]);required=np.array([bool(code&bit) for bit,a,up in BITS])
        if np.all(touch[i,required]):continue
        ids=np.flatnonzero(labels==labels[i]);shift=np.zeros(2);possible=True
        for bit,a,upper in BITS:
            if not code&bit:continue
            # A rigid component cannot expose an interior member at its edge.
            if upper:
                possible&=bool(np.max(x[ids,a]+x[ids,a+2])<=x[i,a]+x[i,a+2]+1e-7)
                shift[a]=hi[a]-x[i,a]-x[i,a+2]
            else:
                possible&=bool(np.min(x[ids,a])>=x[i,a]-1e-7)
                shift[a]=lo[a]-x[i,a]
        if not possible and not flexible:continue
        if p.preplaced[ids].any() and not flexible or p.preplaced[i]:shift[:]=0
        target=x.copy();target[ids,:2]+=shift
        out.append({'kind':'boundary','i':int(i),'ids':ids,'target':target,
                    'rank':float(np.linalg.norm(shift)/scale*(1+len(ids)/p.n))})
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        members=np.flatnonzero(p.constraints[:,3]==g)
        for i in members:
            ids=np.flatnonzero(labels==labels[i])
            if p.preplaced[ids].any() and not flexible or p.preplaced[i]:continue
            for j in members:
                if labels[i]==labels[j]:continue
                for axis in [0,1]:
                    tangent=1-axis;shared=.02*min(x[i,tangent+2],x[j,tangent+2])
                    shift=np.zeros(2)
                    shift[tangent]=np.clip(0,x[j,tangent]-x[i,tangent]-x[i,tangent+2]+shared,
                                           x[j,tangent]+x[j,tangent+2]-x[i,tangent]-shared)
                    for first in [True,False]:
                        shift[axis]=(x[j,axis]-x[i,axis]-x[i,axis+2]) if first else (x[j,axis]+x[j,axis+2]-x[i,axis])
                        target=x.copy();target[ids,:2]+=shift
                        a,b=(int(i),int(j)) if first else (int(j),int(i))
                        out.append({'kind':'contact','i':a,'j':b,'axis':axis,'ids':ids,'target':target,
                            'rank':float(np.linalg.norm(shift)/scale*(1+len(ids)/p.n))})
    out.sort(key=lambda d:(d['rank'],d['kind'],d['i']))
    return out

def coordinated(x,p,labels,groups,proposal,seconds=.025,flexible=False,edge_overrides=None):
    start=time.perf_counter();n=p.n;s=p.scale;z=x/s;wh=z[:,2:];target=proposal['target']/s
    xy=np.arange(2*n).reshape(n,2);lo=2*n+np.arange(2);hi=2*n+2+np.arange(2);nv=2*n+4
    eq=Rows();ineq=Rows();eq.add(xy[p.preplaced].ravel(),1,z[p.preplaced,:2].ravel())
    forest=contact_forest(x,p) if flexible else []
    if not flexible:
        for ids in groups:
            for a in [0,1]:
                eq.add(np.column_stack((xy[ids[1:],a],np.full(len(ids)-1,xy[ids[0],a]))),[1,-1],z[ids[1:],a]-z[ids[0],a])
    else:
        for i,j,a in forest:
            b=1-a;overlap=min(z[i,b]+wh[i,b],z[j,b]+wh[j,b])-max(z[i,b],z[j,b])
            shared=min(.02*min(wh[i,b],wh[j,b]),overlap/2)
            eq.add([[xy[j,a],xy[i,a]]],[1,-1],wh[i,a])
            ineq.add([[xy[i,b],xy[j,b]],[xy[j,b],xy[i,b]]],[1,-1],[wh[j,b]-shared,wh[i,b]-shared])
    ineq.add(np.arange(nv),-1,0);axes=np.tile(np.arange(2),n)
    ineq.add(np.column_stack((lo[axes],xy.ravel())),[1,-1],0)
    ineq.add(np.column_stack((xy.ravel(),hi[axes])),[1,-1],-wh.ravel())
    touching,_,_=touches(x)
    for column,(bit,a,upper) in enumerate(BITS):
        required=(p.constraints[:,4].astype(int)&bit)!=0
        preserved=required&touching[:,column]
        if proposal['kind']=='boundary' and int(p.constraints[proposal['i'],4])&bit:preserved[proposal['i']]=True
        ids=np.flatnonzero(preserved);endpoint=hi[a] if upper else lo[a]
        eq.add(np.column_stack((xy[ids,a],np.full(len(ids),endpoint))),[1,-1],-wh[ids,a] if upper else 0)
    full=topology(x,p.constraints,array=True);changed=topology(proposal['target'],p.constraints,mode=1,array=True)
    involved=np.isin(full[:,:2],proposal['ids']).any(1);full[involved]=changed[involved]
    if not flexible:full=full[labels[full[:,0]]!=labels[full[:,1]]]
    else:
        for i,j,a in forest:
            selected=((full[:,0]==i)&(full[:,1]==j))|((full[:,0]==j)&(full[:,1]==i))
            full[selected]=[i,j,a]
    if proposal['kind']=='contact':
        i,j,a=proposal['i'],proposal['j'],proposal['axis'];b=1-a
        # The enforced new contact chooses the separating direction explicitly.
        selected=((full[:,0]==i)&(full[:,1]==j))|((full[:,0]==j)&(full[:,1]==i))
        full[selected]=[i,j,a]
        eq.add([[xy[j,a],xy[i,a]]],[1,-1],wh[i,a])
        shared=.02*min(wh[i,b],wh[j,b])
        ineq.add([[xy[i,b],xy[j,b]],[xy[j,b],xy[i,b]]],[1,-1],[wh[j,b]-shared,wh[i,b]-shared])
    for pair,edge in (edge_overrides or {}).items():
        use=np.isin(full[:,0],pair)&np.isin(full[:,1],pair);full[use]=edge
    edges=np.asarray(reduce_edges(full,n),dtype=int).reshape(-1,3)
    edge_start=ineq.count
    if len(edges):
        i,j,a=edges.T;ineq.add(np.column_stack((xy[i,a],xy[j,a])),[1,-1],-wh[i,a])
    costs=np.zeros(nv);costs[xy]=-2*target[:,:2]/n;costs[lo]=-.01;costs[hi]=.01
    hessian=diags(np.r_[np.full(2*n,2/n),np.zeros(4)],format='csc')
    ii=[];jj=[];vv=[];rhs=[];offset=0
    for block in [eq,ineq]:
        if block.count:ii.append(np.concatenate(block.i)+offset);jj.extend(block.j);vv.extend(block.v);rhs.extend(block.b)
        offset+=block.count
    matrix=coo_matrix((np.concatenate(vv),(np.concatenate(ii),np.concatenate(jj))),shape=(offset,nv)).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=80
    settings.time_limit=max(.003,seconds-(time.perf_counter()-start));settings.tol_feas=1e-10
    settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
    solver=clarabel.DefaultSolver(hessian,costs,matrix,np.concatenate(rhs),
        [clarabel.ZeroConeT(eq.count),clarabel.NonnegativeConeT(ineq.count)],settings)
    result=solver.solve();stats={'status':str(result.status),'seconds':time.perf_counter()-start,'native_seconds':result.solve_time}
    if 'Infeasible' in str(result.status) and len(edges):
        dual=np.abs(np.asarray(result.z)[eq.count+edge_start:eq.count+edge_start+len(edges)])
        protected={tuple(sorted((i,j))) for i,j,a in forest}
        if proposal['kind']=='contact':protected.add(tuple(sorted((proposal['i'],proposal['j']))))
        ranked=[int(k) for k in np.argsort(-dual) if tuple(sorted(edges[k,:2])) not in protected]
        stats['conflicting_edges']=[edges[k].tolist() for k in ranked[:3]]
    if str(result.status) not in ['Solved','AlmostSolved','MaxTime'] or result.r_prim>1e-7:return None,stats
    out=x.copy();out[:,:2]=np.asarray(result.x)[xy]*s;out[p.preplaced]=p.specs[p.preplaced]
    return out,stats

def shape_coordinated(x,p,proposal,seconds,edge_overrides=None):
    from .fast_geometry import step
    forest=contact_forest(x,p);touch,_,_=touches(x);sides=[]
    for i in np.flatnonzero(p.constraints[:,4]):
        for k,(bit,a,upper) in enumerate(BITS):
            if p.constraints[i,4]&bit and (touch[i,k] or (proposal['kind']=='boundary' and i==proposal['i'])):sides.append((int(i),bit))
    full=topology(x,p.constraints,array=True);changed=topology(proposal['target'],p.constraints,mode=1,array=True)
    involved=np.isin(full[:,:2],proposal['ids']).any(1);full[involved]=changed[involved]
    if proposal['kind']=='contact':forest.append((proposal['i'],proposal['j'],proposal['axis']))
    for i,j,a in forest:
        selected=((full[:,0]==i)&(full[:,1]==j))|((full[:,0]==j)&(full[:,1]==i));full[selected]=[i,j,a]
    for pair,edge in (edge_overrides or {}).items():
        use=np.isin(full[:,0],pair)&np.isin(full[:,1],pair);full[use]=edge
    return step(p,x,reduce_edges(full,p.n),seconds=seconds,reweighted=False,conservative=True,
                full_edges=full,elastic=0.,hard_sides=sides,hard_contacts=forest,threads=1)

def repair(x,p,seconds=.25,max_attempts=24,max_accept=4,flexible=False,rewire=False,shape=False,compact=False):
    start=time.perf_counter();x=x.copy();obj=Objective(p);trace=[];accepted=0;attempts=0
    assert hard_valid(x,p)
    while accepted<max_accept and attempts<max_attempts and time.perf_counter()-start<seconds:
        old_soft=sum(obj.violations(x));old_area,old_wire=raw_measures(x,p)
        if not old_soft and not compact:break
        labels,groups=components(x,p);best=None
        queue=[(proposal,{}) for proposal in proposals(x,p,labels,groups,flexible=flexible)]
        if compact:
            queue.insert(0,({'kind':'compact','i':-1,'ids':np.empty(0,dtype=int),'target':x.copy(),'rank':0.},{}))
        seen=set()
        while queue:
            proposal,overrides=queue.pop(0)
            left=seconds-(time.perf_counter()-start)
            if left<.008 or attempts>=max_attempts:break
            if shape:candidate,stats=shape_coordinated(x,p,proposal,seconds=min(.06,left),edge_overrides=overrides)
            else:candidate,stats=coordinated(x,p,labels,groups,proposal,seconds=min(.025,left),flexible=flexible,edge_overrides=overrides)
            attempts+=1;record={'kind':proposal['kind'],'i':proposal['i'],'rewired':list(overrides.values()),**stats,'accepted':False}
            if rewire and candidate is None and len(overrides)<2:
                alternatives=[]
                for edge in stats.get('conflicting_edges',[])[:2]:
                    i,j,a=edge[:3]
                    pair=tuple(sorted((i,j)))
                    if pair in overrides:continue
                    axis=1-a;centers=proposal['target'][:,:2]+x[:,2:]/2
                    first,second=(i,j) if centers[i,axis]<=centers[j,axis] else (j,i)
                    edited={**overrides,pair:(first,second,axis)}
                    key=(proposal['kind'],proposal['i'],proposal.get('j'),proposal.get('axis'),tuple(sorted(edited.items())))
                    if key not in seen:seen.add(key);alternatives.append((proposal,edited))
                queue[0:0]=alternatives
            if candidate is not None:
                candidate=canonicalize(candidate,p,tolerance=1e-5)
                if hard_valid(candidate,p):
                    soft=sum(obj.violations(candidate));area,wire=raw_measures(candidate,p)
                    bound=max(1.,area/max(old_area,1e-15),wire/max(old_wire,1e-15))*math.exp(2*(soft-old_soft)/obj.nsoft)
                    record.update(soft_before=old_soft,soft_after=soft,certificate_bound=bound)
                    pareto=(area<=old_area*(1+1e-12) and wire<=old_wire*(1+1e-12) and (area<old_area*(1-1e-5) or wire<old_wire*(1-1e-5)))
                    improves=(soft<old_soft and bound<1-1e-10) or (soft<=old_soft and bound<=1+1e-12 and pareto)
                    if improves and (best is None or bound<best[0]):
                        best=(bound,candidate,record)
            trace.append(record)
            # A strict public certificate permits committing immediately.
            if best is not None:break
        if best is None:break
        best[2]['accepted']=True;x=best[1];accepted+=1
    return x,{'seconds':time.perf_counter()-start,'accepted':accepted,'attempts':attempts,'trace':trace,
              'soft_after':list(obj.violations(x)),'certificate_scope':'neutral Q, public geometry and soft counts; fixed/preplaced hard constraints preserved'}
