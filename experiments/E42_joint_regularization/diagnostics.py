"""Continuous draft diagnostics, independent of model selection and repair."""
import numpy as np
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks


def geometry_checks(positions, inputs):
    n, areas, _, _, _, constraints, specs = inputs
    areas = np.asarray(areas, dtype=float)
    positions = np.asarray(positions, dtype=float)[:n].copy()
    c, specs = np.asarray(constraints), np.asarray(specs)
    locked = c[:, :2].any(1)
    positions[locked, 2:] = specs[locked, 2:]
    positions[c[:, 1] != 0] = specs[c[:, 1] != 0]
    checks = independent_hard_checks(positions.tolist(), areas, c,
                                     public_hard_reference(areas, c, specs))
    checks['negative_coordinate_blocks'] = np.flatnonzero((positions[:, :2] < 0).any(1)).tolist()
    checks['is_feasible'] = checks['is_feasible'] and not checks['negative_coordinate_blocks']
    low = np.maximum(positions[:, None, :2], positions[None, :, :2])
    high = np.minimum((positions[:, :2]+positions[:, 2:])[:, None],
                      (positions[:, :2]+positions[:, 2:])[None, :])
    penetration = np.maximum(high-low, 0)
    intersection = penetration.prod(-1)
    i, j = np.triu_indices(n, 1)
    scale = np.sqrt(np.asarray(areas, dtype=float))
    local = (scale[i]+scale[j])/2
    normalized = np.min(penetration[i, j], axis=1)/np.maximum(local, 1e-12)
    parent = np.arange(n)
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for a, b in checks['overlap_pairs']:
        parent[find(a)] = find(b)
    counts = np.bincount([find(k) for k in range(n)], minlength=n)
    largest = int(counts.max()) if checks['overlap_pairs'] else 0
    return positions, checks, {
        'overlap_pairs': len(checks['overlap_pairs']),
        'overlap_area_fraction': float(intersection[i, j].sum()/np.sum(areas)),
        'penetration_squared_per_block': float(np.sum(normalized**2)/n),
        'max_local_penetration': float(normalized.max(initial=0)),
        'largest_conflict_cluster': largest,
        'largest_conflict_cluster_fraction': largest/n,
    }


def distribution(records, keys):
    result = {}
    for key in keys:
        values = np.asarray([r[key] for r in records], dtype=float)
        result[key] = {'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)),
                       'max': float(values.max())}
    return result
