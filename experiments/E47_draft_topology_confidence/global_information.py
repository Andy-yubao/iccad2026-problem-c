"""Explicitly privileged global-topology sufficiency ablations.

Only relation triples and confidence/adjacency labels survive construction.
Continuous iteration always starts from the unchanged full model draft.
"""
from dataclasses import replace
import numpy as np
from .fast_geometry import topology,contact_tree


def acyclic(edges,n):
    for axis in (0,1):
        adj=[[] for _ in range(n)];degree=np.zeros(n,int)
        for a,b,_ in edges[edges[:,2]==axis]:adj[a].append(b);degree[b]+=1
        todo=list(np.flatnonzero(degree==0));visited=0
        while todo:
            a=todo.pop();visited+=1
            for b in adj[a]:
                degree[b]-=1
                if degree[b]==0:todo.append(b)
        if visited!=n:return False
    return True


def contact_forest(draft,p,edges,g):
    """Preserve true in-group trees first, then connect across groups.

    This is a geometric forest, not a fabricated public grouping constraint.
    """
    parent=np.arange(p.n);selected=[]
    def root(a):
        while parent[a]!=a:a=parent[a]
        return a
    def add(edge):
        a,b,axis=map(int,edge);ra,rb=root(a),root(b)
        if ra!=rb and g.contact_confidence[a,b]>=.9:
            parent[ra]=rb;selected.append((a,b,axis))
    for edge in contact_tree(draft,p,edges,g):add(edge)
    a,b,axis=edges.T;t=1-axis
    gap=abs(draft[b,axis]-draft[a,axis]-draft[a,axis+2])
    tangent=np.maximum.reduce((np.zeros(len(a)),draft[a,t]-draft[b,t]-draft[b,t+2],draft[b,t]-draft[a,t]-draft[a,t+2]))
    for at in np.lexsort((b,a,gap+2*tangent)):add(edges[at])
    return np.asarray(selected,int).reshape(-1,3)


def build(cls,draft,p,kind,witness,seed):
    base=cls.build(draft,p,'uniform' if kind=='draft_static' else 'oracle_contact',witness,seed)
    raw=topology(draft,p.constraints,array=True);events=[]
    if kind=='draft_static':
        trust=np.ones(len(raw));selected=raw.copy()
    else:
        proposed=topology(witness,p.constraints,array=True);a,b,axis=raw.T
        valid=witness[b,axis]-witness[a,axis]-witness[a,axis+2]>=-1e-7
        trust=np.where(valid,.98,.02)
        if kind=='oracle_full':selected=proposed.copy()
        else:
            if kind=='shuffle_hybrid':
                valid=valid[np.random.default_rng(seed).permutation(len(valid))];trust=np.where(valid,.98,.02)
            selected=np.where(valid[:,None],raw,proposed)
        if not acyclic(selected,p.n):
            events.append({'reason':'global_cycle_fallback','pair':None});selected=raw.copy()
    commit=kind=='oracle_hybrid_contacts'
    g=replace(base,metric_enabled=False,contact_enabled=commit,priority=commit,commit=commit,
        static_edges=selected,topology_confidence=trust,global_mode=kind,events=events)
    forest=contact_forest(draft,p,selected,g) if commit else None
    for arr in (selected,trust,forest):
        if arr is not None:arr.flags.writeable=False
    return replace(g,global_edges=forest)
