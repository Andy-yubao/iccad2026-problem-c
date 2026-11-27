"""Output geometry summaries for sparse-anchor diagnosis; no fitting or solver."""
import numpy as np
from scipy.sparse.csgraph import shortest_path


def coverage_label(p):
    return 'low' if p <= .3 else ('high' if p >= .8 else 'middle')


def spatial_metrics(prediction, reference, mutable):
    p = np.asarray(prediction)[mutable]; r = np.asarray(reference)[mutable]
    if len(p) < 2:
        return {}
    p = p-p.mean(0); r = r-r.mean(0)
    rr = np.square(r).sum(); pp = np.square(p).sum()
    alpha = (p*r).sum()/max(rr,1e-12)
    return {'spread_ratio': float(np.sqrt(pp/max(rr,1e-12))),
            'fit_reference_scale': float(alpha),
            'reference_rms_spread': float(np.sqrt(rr/len(r))),
            'predicted_rms_spread': float(np.sqrt(pp/len(p))),
            'centered_mse': float(np.square(p-r).sum()/len(p)),
            'residual_after_reference_scale_mse': float(np.square(p-alpha*r).sum()/len(p))}


def graph_node_metrics(payload):
    member, inputs, reference, positions = payload
    n, areas, b2b, p2b, pins, constraints, specs = inputs
    c = np.asarray(constraints); s = np.asarray(specs); pre = c[:,1] != 0; mutable = ~pre
    area = np.where(c[:,:2].any(1),s[:,2:].prod(1),areas); size=np.sqrt(area)
    adj = np.zeros((n,n),bool); pins_connected=np.zeros(n,bool)
    for i,j,w in np.asarray(b2b):
        if w>0:adj[int(i),int(j)]=adj[int(j),int(i)]=True
    for pi,bi,w in np.asarray(p2b):
        if w>0:pins_connected[int(bi)]=True
    dist=shortest_path(adj,directed=False,unweighted=True)
    hop=dist[:,pins_connected].min(1) if pins_connected.any() else np.full(n,np.inf)
    rp=reference[:,:2]+reference[:,2:]/2; pp=positions[:,:2]+positions[:,2:]/2
    local_error=np.linalg.norm(pp-rp,axis=1)/size
    groups={}
    for label,selection in (('pin',pins_connected),('no_pin',~pins_connected),
                            ('hop1',hop==1),('hop2plus',np.isfinite(hop)&(hop>=2)),
                            ('unreachable',~np.isfinite(hop))):
        ids=selection&mutable
        groups[label]={'count':int(ids.sum()),'mean_local_error':float(local_error[ids].mean()) if ids.any() else None,
                       **spatial_metrics(pp,rp,ids)}
    all_features=np.column_stack((area,c))
    exact_duplicates=n-len(np.unique(all_features,axis=0))
    return {'member':member,'pin_coverage':float(pins_connected.mean()),
            'density':float(adj.sum()/max(n*(n-1),1)),
            'node_groups':groups,'free_node_spatial':spatial_metrics(pp,rp,mutable),
            'duplicate_area_constraint_rows':int(exact_duplicates),
            'note':'Duplicate area/constraint rows are not duplicate complete graph inputs.'}


def cosine(a,b):
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den>1e-15 else None


def mean_rows(rows,field):
    vals=[r[field] for r in rows if field in r and r[field] is not None]
    return float(np.mean(vals)) if vals else None
