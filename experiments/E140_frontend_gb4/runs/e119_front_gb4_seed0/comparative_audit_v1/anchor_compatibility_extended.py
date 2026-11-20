"""Necessary public-anchor compatibility checks on initial separation graphs.

These are algebraic witnesses for the fixed graph, not infeasibility claims
about the original task. No optimizer, projection, packing or geometry update.
"""
import heapq
import numpy as np
from separation import topology


def examine(draft, inputs, supplied_edges=None):
    n,areas,_,_,_,flags,specs=inputs
    flags=np.asarray(flags,int);specs=np.asarray(specs,float);draft=np.asarray(draft,float)
    locked=(flags[:,0]!=0)|(flags[:,1]!=0)
    pre=np.flatnonzero(flags[:,1]!=0)
    ratio=draft[:,2]/draft[:,3]
    dims=np.stack((np.sqrt(areas*ratio),np.sqrt(areas/ratio)),-1)
    dims[locked]=specs[locked,2:]
    seed=np.column_stack((draft[:,:2],dims))
    edges=topology(seed,flags,array=True) if supplied_edges is None else np.asarray(supplied_edges)
    minimum=np.where(locked[:,None],dims,0.)
    tolerance=1e-7*np.sqrt(np.sum(areas))
    result=dict(direct_PP_conflicts=[],anchor_path_conflicts=[],nonnegative_path_conflicts=[],area_trap_conflicts=[])
    lower_bounds=[];upper_bounds=[];lower_chains=[];upper_chains=[]
    for u,v,axis in edges:
        if flags[u,1] and flags[v,1]:
            required=float(specs[u,axis]+dims[u,axis])
            available=float(specs[v,axis])
            if required>available+tolerance:
                result['direct_PP_conflicts'].append(dict(axis=int(axis),first=int(u),second=int(v),
                    required_second_coordinate_at_least=required,actual_second_coordinate=available,deficit=required-available))
    for axis in (0,1):
        adj=[[] for _ in range(n)];degree=np.zeros(n,int)
        for u,v,_ in edges[edges[:,2]==axis]:adj[u].append(int(v));degree[v]+=1
        queue=np.flatnonzero(degree==0).tolist();heapq.heapify(queue);order=[]
        while queue:
            u=heapq.heappop(queue);order.append(u)
            for v in adj[u]:
                degree[v]-=1
                if degree[v]==0:heapq.heappush(queue,v)
        assert len(order)==n,'Center-ordered separation graph must be acyclic'
        def longest(start=None):
            values=np.zeros(n) if start is None else np.full(n,-np.inf)
            parent=np.full(n,-1,int)
            if start is not None:values[start]=0.
            for u in order:
                if not np.isfinite(values[u]):continue
                candidate=values[u]+minimum[u,axis]
                for v in adj[u]:
                    if candidate>values[v]:values[v]=candidate;parent[v]=u
            return values,parent
        def chain(parent,end):
            answer=[int(end)]
            while parent[answer[-1]]>=0:
                answer.append(int(parent[answer[-1]]));assert len(answer)<=n
            return answer[::-1]
        for first in pre:
            values,parent=longest(int(first))
            for second in pre:
                if second==first or not np.isfinite(values[second]):continue
                available=float(specs[second,axis]-specs[first,axis])
                if values[second]>available+tolerance:
                    result['anchor_path_conflicts'].append(dict(axis=axis,first=int(first),second=int(second),
                        path=chain(parent,int(second)),minimum_required_span=float(values[second]),available_span=available,
                        deficit=float(values[second]-available)))
        values,parent=longest()
        for end in pre:
            if values[end]>specs[end,axis]+tolerance:
                result['nonnegative_path_conflicts'].append(dict(axis=axis,last=int(end),path=chain(parent,int(end)),
                    minimum_required_coordinate=float(values[end]),fixed_coordinate=float(specs[end,axis]),
                    deficit=float(values[end]-specs[end,axis])))
        lower=np.zeros(n);lp=np.full(n,-1,int)
        upper=np.full(n,np.inf);up=np.full(n,-1,int)
        for u in order:
            if flags[u,1] and specs[u,axis]>lower[u]:lower[u]=specs[u,axis];lp[u]=-1
            for v in adj[u]:
                candidate=lower[u]+minimum[u,axis]
                if candidate>lower[v]:lower[v]=candidate;lp[v]=u
        for u in order[::-1]:
            if flags[u,1]:upper[u]=specs[u,axis]
            for v in adj[u]:
                candidate=upper[v]-minimum[u,axis]
                if candidate<upper[u]:upper[u]=candidate;up[u]=v
        lower_bounds.append(lower);upper_bounds.append(upper)
        lower_chains.append([chain(lp,u) for u in range(n)])
        paths=[]
        for u in range(n):
            path=[u]
            while up[path[-1]]>=0:path.append(int(up[path[-1]]));assert len(path)<=n
            paths.append(path)
        upper_chains.append(paths)
    for i in np.flatnonzero(~locked):
        capacity=np.array([upper_bounds[a][i]-lower_bounds[a][i] for a in (0,1)])
        if np.isfinite(capacity).all() and float(np.maximum(capacity,0).prod()) < .99*areas[i]-tolerance**2:
            result['area_trap_conflicts'].append(dict(module=int(i),maximum_width_height=capacity.tolist(),
                maximum_area=float(np.maximum(capacity,0).prod()),minimum_legal_area=float(.99*areas[i]),
                lower_bounds=[float(lower_bounds[a][i]) for a in (0,1)],
                upper_bounds=[float(upper_bounds[a][i]) for a in (0,1)],
                lower_paths=[lower_chains[a][i] for a in (0,1)],upper_paths=[upper_chains[a][i] for a in (0,1)]))
    result['any_conflict']=any(result[name] for name in ('direct_PP_conflicts','anchor_path_conflicts','nonnegative_path_conflicts','area_trap_conflicts'))
    return result
