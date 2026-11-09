"""Public and explicitly label-assisted explanatory variables; no solver or model."""
import os
import numpy as np


def components_count(adjacency):
    n = len(adjacency)
    seen = np.zeros(n, bool)
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1; seen[start] = True; stack = [start]
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(adjacency[node] & ~seen):
                seen[nxt] = True; stack.append(nxt)
    return count


def extract(payload):
    member, inputs, reference, public_features = payload
    n, area, b2b, p2b, pins, constraints, specs = inputs
    n = int(n); area = np.asarray(area, float); c = np.asarray(constraints, int); specs = np.asarray(specs, float)
    fixed = c[:, :2].any(1); pre = c[:, 1] != 0
    effective = np.where(fixed, specs[:, 2:].prod(1), area)
    local = np.sqrt(effective); scale = np.sqrt(effective.sum())
    share = effective/effective.sum(); ordered = np.sort(effective)
    gini = (2*np.arange(1, n+1)-n-1) @ ordered/(n*ordered.sum())
    b2b = np.asarray(b2b, float); b2b = b2b[~(b2b == -1).all(1)]
    p2b = np.asarray(p2b, float); p2b = p2b[~(p2b == -1).all(1)]
    adjacency = np.zeros((n, n)); pinweight = np.zeros(n)
    if len(b2b):
        i, j = b2b[:, :2].astype(int).T
        np.add.at(adjacency, (i, j), b2b[:, 2]); np.add.at(adjacency, (j, i), b2b[:, 2])
    if len(p2b):
        np.add.at(pinweight, p2b[:, 1].astype(int), p2b[:, 2])
    edges = adjacency > 0; np.fill_diagonal(edges, False)
    degree = edges.sum(1); weightdegree = adjacency.sum(1)
    counts = {}
    for name, column in (('mib', 2), ('group', 3)):
        ids, sizes = np.unique(c[c[:, column] > 0, column], return_counts=True)
        counts.update({name+'_fraction': float((c[:, column] > 0).mean()),
                       name+'_groups_per_node': len(ids)/n,
                       name+'_max_size_fraction': float(sizes.max(initial=0)/n)})
    boundary_bits = np.array([[(flag & bit) != 0 for bit in (1, 2, 4, 8)] for flag in c[:, 4]])
    i, j = np.triu_indices(n, 1)
    cross = edges[i, j] & (c[i, 3] != c[j, 3])
    out = {'member': member, 'worker': member[0].split('/')[0], 'n': n,
           'log_n': float(np.log(n)), 'log_n_squared': float(np.log(n)**2),
           'area_cv': float(effective.std()/effective.mean()), 'area_gini': float(gini),
           'log_area_ratio': float(np.log(effective.max()/effective.min())),
           'max_area_share': float(share.max()), 'effective_module_fraction': float(1/(share @ share)/n),
           'fixed_fraction': float(fixed.mean()), 'preplaced_fraction': float(pre.mean()),
           'fixed_area_share': float(share[fixed].sum()), 'preplaced_area_share': float(share[pre].sum()),
           'boundary_fraction': float((c[:, 4] > 0).mean()), 'boundary_edges_per_node': float(boundary_bits.sum()/n),
           'connection_density': float(edges.sum()/max(n*(n-1), 1)),
           'mean_degree': float(degree.mean()), 'degree_cv': float(degree.std()/max(degree.mean(), 1e-12)),
           'weighted_degree_cv': float(weightdegree.std()/max(weightdegree.mean(), 1e-12)),
           'isolated_fraction': float((degree == 0).mean()),
           'graph_components_fraction': components_count(edges)/n,
           'cross_group_edge_fraction': float(cross.sum()/max(edges[i, j].sum(), 1)),
           'pin_edges_per_node': len(p2b)/n, 'pin_count_per_node': len(pins)/n,
           'pin_connected_fraction': float((pinweight > 0).mean()),
           'pin_weight_cv': float(pinweight.std()/max(pinweight.mean(), 1e-12)), **counts}
    unique_features = np.unique(np.round(np.asarray(public_features), 6), axis=0)
    out['duplicate_node_feature_fraction'] = 1-len(unique_features)/n
    # Label-assisted descriptors below are explanations, never prediction-time inputs.
    r = np.asarray(reference, float)/scale
    centers = r[:, :2]+r[:, 2:]/2
    hi = (r[:, :2]+r[:, 2:]).max(0); lo = r[:, :2].min(0); extent = hi-lo
    pairscale = (local[:, None]+local[None, :])/(2*scale)
    delta = centers[:, None]-centers[None]
    margin = abs(delta)-(r[:, None, 2:]+r[None, :, 2:])/2
    mutable = ~pre[:, None] | ~pre[None, :]
    pairs = ~np.eye(n, dtype=bool) & mutable
    near = (np.linalg.norm(np.maximum(margin, 0), axis=-1) <= .25*pairscale) & pairs
    same_group = (c[:, 3, None] == c[None, :, 3]) & (c[:, 3, None] > 0)
    contacts = (((abs(margin[..., 0]) < 1e-6) & (margin[..., 1] < -1e-6)) |
                ((abs(margin[..., 1]) < 1e-6) & (margin[..., 0] < -1e-6))) & same_group & pairs
    ratios = np.log(r[:, 2]/r[:, 3])
    out.update(ref_bbox_log_aspect=float(abs(np.log(extent[0]/extent[1]))),
               ref_fill_fraction=float(r[:, 2:].prod(-1).sum()/extent.prod()),
               ref_shape_std=float(ratios.std()), ref_abs_log_aspect_mean=float(abs(ratios).mean()),
               ref_near_pairs_per_node=float(near.sum()/n),
               ref_contact_pairs_per_node=float(contacts.sum()/n),
               ref_contact_target_present=float(contacts.any()),
               ref_area_transform_log_mismatch=float(abs(np.log(effective/(reference[:, 2:].prod(-1)))).mean()),
               extraction_pid=os.getpid())
    return out


PUBLIC = ['log_n', 'log_n_squared', 'area_cv', 'area_gini', 'log_area_ratio', 'max_area_share',
          'effective_module_fraction', 'fixed_fraction', 'preplaced_fraction', 'fixed_area_share',
          'preplaced_area_share', 'boundary_fraction', 'boundary_edges_per_node', 'connection_density',
          'mean_degree', 'degree_cv', 'weighted_degree_cv', 'isolated_fraction', 'graph_components_fraction',
          'cross_group_edge_fraction', 'pin_edges_per_node', 'pin_count_per_node', 'pin_connected_fraction',
          'pin_weight_cv', 'mib_fraction', 'mib_groups_per_node', 'mib_max_size_fraction', 'group_fraction',
          'group_groups_per_node', 'group_max_size_fraction', 'duplicate_node_feature_fraction']
REFERENCE = ['ref_bbox_log_aspect', 'ref_fill_fraction', 'ref_shape_std', 'ref_abs_log_aspect_mean',
             'ref_near_pairs_per_node', 'ref_contact_pairs_per_node', 'ref_contact_target_present',
             'ref_area_transform_log_mismatch']
