"""Necessary DAG + public-anchor + area bounds; no placement optimizer."""
import heapq
import numpy as np


def propagate(edges,areas,flags,specs,max_iterations=200):
    edges=np.asarray(edges,int);areas=np.asarray(areas,float);n=len(areas)
    flags=np.asarray(flags,int);specs=np.asarray(specs,float)
    locked=(flags[:,0]!=0)|(flags[:,1]!=0);pre=flags[:,1]!=0
    minimum=np.where(locked[:,None],specs[:,2:],0.)
    tol=1e-7*np.sqrt(areas.sum())
    adj=[];order=[]
    for axis in (0,1):
        rows=[[] for _ in range(n)];degree=np.zeros(n,int)
        for u,v,a in edges[edges[:,2]==axis]:rows[u].append(int(v));degree[v]+=1
        queue=np.flatnonzero(degree==0).tolist();heapq.heapify(queue);sequence=[]
        while queue:
            u=heapq.heappop(queue);sequence.append(u)
            for v in rows[u]:
                degree[v]-=1
                if degree[v]==0:heapq.heappush(queue,v)
        if len(sequence)!=n:return dict(conflict=True,reason='positive-dimension directed cycle',axis=axis)
        adj.append(rows);order.append(sequence)
    history=[]
    for iteration in range(max_iterations):
        caps=np.full((n,2),np.inf);lower_all=[];right_all=[]
        for axis in (0,1):
            lower=np.zeros(n);upper=np.full(n,np.inf);lp=np.full(n,-1,int)
            for u in order[axis]:
                if pre[u] and specs[u,axis]>lower[u]:lower[u]=specs[u,axis];lp[u]=-1
                for v in adj[axis][u]:
                    value=lower[u]+minimum[u,axis]
                    if value>lower[v]:lower[v]=value;lp[v]=u
            for u in order[axis][::-1]:
                if pre[u]:upper[u]=specs[u,axis]
                for v in adj[axis][u]:upper[u]=min(upper[u],upper[v]-minimum[u,axis])
            lower_all.append(lower);right_all.append(upper+minimum[:,axis]);caps[:,axis]=upper+minimum[:,axis]-lower
            bad=pre&((lower>specs[:,axis]+tol)|(upper<specs[:,axis]-tol))
            if bad.any():
                witnesses=[]
                for v in np.flatnonzero(bad):
                    path=[int(v)]
                    while lp[path[-1]]>=0:path.append(int(lp[path[-1]]));assert len(path)<=n
                    path=path[::-1]
                    witnesses.append(dict(module=int(v),fixed_coordinate=float(specs[v,axis]),lower_bound=float(lower[v]),
                        upper_bound=float(upper[v]),lower_path=path,
                        path_dimension_lower_bounds=[float(minimum[u,axis]) for u in path[:-1]]))
                return dict(conflict=True,reason='propagated lower/upper bounds contradict fixed anchor',axis=axis,
                    modules=np.flatnonzero(bad).tolist(),anchor_witnesses=witnesses,iterations=iteration+1,history=history,
                    minimum_width_height=minimum.tolist(),caps=caps.tolist())
        for u in np.flatnonzero(~locked):
            if (caps[u]<=0).any() or np.isfinite(caps[u]).all() and caps[u].prod()<.99*areas[u]-tol**2:
                return dict(conflict=True,reason='available width-height product smaller than minimum legal area',module=int(u),
                    maximum_width_height=caps[u].tolist(),maximum_area=float(np.maximum(caps[u],0).prod()),minimum_area=float(.99*areas[u]),
                    iterations=iteration+1,history=history)
        updated=minimum.copy()
        for axis in (0,1):
            other=1-axis;eligible=(~locked)&np.isfinite(caps[:,other])&(caps[:,other]>0)
            updated[eligible,axis]=np.maximum(updated[eligible,axis],.99*areas[eligible]/caps[eligible,other])
        history.append(dict(iteration=iteration+1,module100_minimum=updated[100].tolist() if n>100 else None,
            module100_capacity=caps[100].tolist() if n>100 else None,
            module18_minimum=updated[18].tolist() if n>18 else None))
        if np.max(updated-minimum)<=tol:return dict(conflict=False,reason='no contradiction detected; not a feasibility proof',iterations=iteration+1,history=history)
        minimum=updated
    return dict(conflict=False,reason='iteration cap; no certificate',iterations=max_iterations,history=history)
