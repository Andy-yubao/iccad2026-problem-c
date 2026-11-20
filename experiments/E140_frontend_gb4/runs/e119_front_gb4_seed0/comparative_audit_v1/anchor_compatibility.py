"""Necessary public-anchor compatibility checks on initial separation graphs.

These are algebraic witnesses for the fixed graph, not infeasibility claims
about the original task. No optimizer, projection, packing or geometry update.
"""
import heapq
import numpy as np
from separation import topology


def examine(draft, inputs):
    n,areas,_,_,_,flags,specs=inputs
    flags=np.asarray(flags,int);specs=np.asarray(specs,float);draft=np.asarray(draft,float)
    locked=(flags[:,0]!=0)|(flags[:,1]!=0)
    pre=np.flatnonzero(flags[:,1]!=0)
    ratio=draft[:,2]/draft[:,3]
    dims=np.stack((np.sqrt(areas*ratio),np.sqrt(areas/ratio)),-1)
    dims[locked]=specs[locked,2:]
    seed=np.column_stack((draft[:,:2],dims))
    edges=topology(seed,flags,array=True)
    minimum=np.where(locked[:,None],dims,0.)
    tolerance=1e-7*np.sqrt(np.sum(areas))
    result=dict(direct_PP_conflicts=[],anchor_path_conflicts=[],nonnegative_path_conflicts=[])
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
    result['any_conflict']=any(result[name] for name in ('direct_PP_conflicts','anchor_path_conflicts','nonnegative_path_conflicts'))
    return result
