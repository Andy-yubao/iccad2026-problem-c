"""Deterministic public-only structural roles and connected-port tokens."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,shortest_path

NAMES=('degree_fraction','degree_log','neighbor_degree_mean','neighbor_degree_std','component_fraction',
    'reachable2_fraction','reachable4_fraction','mean_hop_asinh','isolated',
    'nearest_P_hop','has_reachable_P','nearest_port_hop','has_reachable_port',
    'direct_port_fraction','graph_density','node_count_log')

def build(g):
    # Explicit whitelist: references, targets and their metrics are never accessed.
    n=int(g['n']);wire=g['wire_b2b'].numpy().astype(np.float32)*2
    wire=wire.copy();np.fill_diagonal(wire,0)
    adj=wire>0;degree=adj.sum(1).astype(np.float32)
    coordinates=g['pin_coordinates'].numpy();valid=(g['pin_valid']&(g['pin_weights']>0)).numpy()
    weights=g['pin_weights'].numpy()
    ports=np.unique(coordinates[valid],axis=0).astype(np.float32) if valid.any() else np.zeros((0,2),np.float32)
    m=len(ports);lookup={tuple(p):i for i,p in enumerate(ports)}
    edges=np.zeros((n,m),np.float32)
    for i in range(n):
        for j in np.flatnonzero(valid[i]):edges[i,lookup[tuple(coordinates[i,j])]]+=weights[i,j]
    combined=np.zeros((n+m,n+m),np.float32);combined[:n,:n]=wire
    combined[:n,n:]=edges;combined[n:,:n]=edges.T
    distance=shortest_path(csr_matrix(combined>0),directed=False,unweighted=True,method='D',indices=np.arange(n))
    hops=np.where(np.isfinite(distance),distance,-1).astype(np.int16)
    _,labels=connected_components(csr_matrix(adj),directed=False)
    component=np.bincount(labels)[labels]/n
    mean_neighbor=(adj@degree)/np.maximum(degree,1)
    var_neighbor=(adj@(degree**2))/np.maximum(degree,1)-mean_neighbor**2
    d=distance[:n,:n];other=~np.eye(n,dtype=bool);reachable=np.isfinite(d)&other
    mean_hop=np.where(reachable,d,0).sum(1)/np.maximum(reachable.sum(1),1)
    def nearest(mask):
        if not np.any(mask):return np.zeros(n),np.zeros(n)
        best=distance[:n,mask].min(1);has=np.isfinite(best)
        return np.arcsinh(np.where(has,best,0)),has.astype(np.float32)
    p_mask=np.r_[g['constraints'].numpy()[:,1]!=0,np.zeros(m,bool)]
    p_dist,p_has=nearest(p_mask);port_dist,port_has=nearest(np.r_[np.zeros(n,bool),np.ones(m,bool)])
    norm=max(n-1,1)
    role=np.stack((degree/norm,np.log1p(degree),mean_neighbor/norm,np.sqrt(np.maximum(var_neighbor,0))/norm,
        component,((d<=2)&reachable).sum(1)/norm,((d<=4)&reachable).sum(1)/norm,np.arcsinh(mean_hop),degree==0,
        p_dist,p_has,port_dist,port_has,(edges>0).sum(1)/max(m,1),
        np.full(n,degree.sum()/(n*norm)),np.full(n,np.log1p(n))),-1).astype(np.float32)
    port_features=np.column_stack((ports,np.log1p(edges.sum(0)),np.log1p((edges>0).sum(0)))).astype(np.float32)
    assert role.shape==(n,16) and port_features.shape==(m,4)
    assert np.isfinite(role).all() and np.isfinite(port_features).all()
    return dict(role_features=role,port_features=port_features,port_edges=edges,hops=hops)
