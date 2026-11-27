"""Target individual public boundary/group failures with local relation changes."""
import time
import numpy as np
from .fast_geometry import topology,reduce_edges,contact_tree,step
from .projection import solve as project

def bad_boundary(x,p):
    lo=x[:,:2].min(0);hi=(x[:,:2]+x[:,2:]).max(0)
    touch=np.column_stack((abs(x[:,0]-lo[0])<1e-6,abs(x[:,0]+x[:,2]-hi[0])<1e-6,
        abs(x[:,1]+x[:,3]-hi[1])<1e-6,abs(x[:,1]-lo[1])<1e-6))
    req=(p.constraints[:,4,None].astype(int)&np.array([1,2,4,8]))!=0
    return np.flatnonzero(np.any(req&~touch,axis=1))

def boundary_edges(x,p,target,variant):
    edges=topology(x,p.constraints,array=True);center=x[:,:2]+x[:,2:]/2;code=int(p.constraints[target,4])
    for k,(i,j,a) in enumerate(edges):
        if target not in(i,j):continue
        for bit,axis,lower in ((1,0,True),(2,0,False),(4,1,False),(8,1,True)):
            if not(code&bit) or a!=axis:continue
            blocked=(j==target) if lower else(i==target)
            if blocked:
                other=j if i==target else i;b=1-a
                first,second=(target,other) if center[target,b]<=center[other,b] else(other,target)
                if variant%2:first,second=second,first
                edges[k]=(first,second,b)
    return edges

def disconnected_pairs(x,p):
    pairs=[]
    for g in np.unique(p.constraints[:,3]):
        if g<=0:continue
        ids=np.flatnonzero(p.constraints[:,3]==g);r=x[ids];parent=list(range(len(ids)))
        def root(a):
            while parent[a]!=a:a=parent[a]
            return a
        overlap=np.minimum((r[:,:2]+r[:,2:])[:,None],(r[:,:2]+r[:,2:])[None])-np.maximum(r[:,:2][:,None],r[:,:2][None])
        touching=((overlap[:,:,0]>=0)&(overlap[:,:,1]>0))|((overlap[:,:,0]>0)&(overlap[:,:,1]>=0))
        for i,j in zip(*np.tril_indices(len(ids),-1)):
            if touching[i,j]:parent[root(i)]=root(j)
        options=[]
        for i,j in zip(*np.tril_indices(len(ids),-1)):
            if root(i)==root(j):continue
            gap=np.maximum(-overlap[i,j],0);options.append((float(gap.sum()),int(ids[i]),int(ids[j])))
        pairs.extend((i,j) for _,i,j in sorted(options)[:2])
    return pairs

def repair(archive,base,deadline,seconds=.7,shape=False):
    stop=min(deadline,time.perf_counter()+seconds);p=archive.p;trace=[]
    x=np.asarray(base['positions']);targets=[('boundary',int(i)) for i in bad_boundary(x,p)]+[('group',pair) for pair in disconnected_pairs(x,p)]
    for kind,target in targets:
        for variant in (0,1):
            left=stop-time.perf_counter()
            if left<.06:return trace
            seed=np.asarray(min(archive.rows,key=lambda r:r['public_quality_cost']+.02*r['distance']['D'])['positions'])
            if kind=='boundary':
                full=boundary_edges(seed,p,target,variant);bt=[target];ct=[]
            else:
                full=topology(seed,p.constraints,array=True);i,j=target;axis=variant;centers=seed[:,:2]+seed[:,2:]/2
                a,b=(i,j) if centers[i,axis]<=centers[j,axis] else(j,i)
                full[max(i,j)*(max(i,j)-1)//2+min(i,j)]=(a,b,axis);bt=[];ct=[(i,j)]
            tree=contact_tree(seed,p,full)
            if ct:
                tree=[t for t in tree if set(t[:2])!=set(ct[0])]+[(a,b,axis)]
            y,info=project(p,seed,reduce_edges(full,p.n),tree,seconds=min(.24,left-.025),boundary=32.,contact=24.,
                anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale),draft_reference=archive.draft,
                distance_weight=.003,engine='highs',boundary_targets=bt,contact_targets=ct,preserve_satisfied=True)
            accepted=archive.accept(y,f'target_{kind}/{target}/{variant}')
            trace.append({'phase':'target_repair','kind':kind,'target':target,'variant':variant,'accepted':accepted is not None,**info})
            if shape and accepted is None and stop-time.perf_counter()>.09:
                y,info=step(p,seed,reduce_edges(full,p.n),seconds=min(.20,stop-time.perf_counter()-.025),full_edges=full,
                    elastic=0.,reweighted=False,conservative=False,boundary=32.,contact=24.,
                    anchors=(archive.obj.area_scale/p.scale**2,archive.draft_wire/p.scale),draft_reference=archive.draft,
                    distance_weight=.005,shape_weight=.02,boundary_targets=bt,contact_targets=ct,tree_override=tree,preserve_soft=True)
                accepted=archive.accept(y,f'target_shape_{kind}/{target}/{variant}')
                trace.append({'phase':'target_shape_repair','kind':kind,'target':target,'variant':variant,'accepted':accepted is not None,**info})
    return trace
