"""Input-only geometry, shared MIB shapes, and a conservative legal incumbent.

Algorithms are local to E44. No other experiment or reference label is imported.
The caller, rather than this solver, owns any mean-area dataset transform.
"""
from dataclasses import dataclass
import math
import numpy as np
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator


@dataclass
class Problem:
    n: int
    areas: np.ndarray
    b2b: np.ndarray
    p2b: np.ndarray
    pins: np.ndarray
    constraints: np.ndarray
    specs: np.ndarray
    scale: float

    @classmethod
    def from_inputs(cls, inputs):
        n = int(inputs[0])
        arrays = [np.asarray(v, dtype=np.float64).copy() for v in inputs[1:]]
        areas, b2b, p2b, pins, constraints, specs = arrays
        if n < 1 or areas.shape != (n,) or (areas <= 0).any():
            raise ValueError('Expected positive unpadded public areas')
        if constraints.shape != (n, 5) or specs.shape != (n, 4):
            raise ValueError('Expected constraints[n,5] and masked specs[n,4]')
        if any(not np.isfinite(v).all() for v in arrays):
            raise ValueError('Nonfinite public input')
        if (constraints < 0).any() or not np.equal(constraints, constraints.astype(int)).all():
            raise ValueError('Constraint flags must be nonnegative integers')
        if (constraints[:, 4] > 15).any():
            raise ValueError('Unknown boundary bitmask')
        locked, pre = constraints[:, :2].any(1), constraints[:, 1] != 0
        if (specs[locked, 2:] <= 0).any() or (specs[pre, :2] < 0).any():
            raise ValueError('Invalid hard dimensions/anchors')
        # Reject accidental unmasked free XY/WH at the public solver boundary.
        if not (specs[~locked] == -1).all() or not (specs[locked & ~pre, :2] == -1).all():
            raise ValueError('Free reference information in solver inputs')
        for edges, counts in ((b2b, (n, n)), (p2b, (len(pins), n))):
            if edges.ndim != 2 or edges.shape[1] != 3:
                raise ValueError('Expected weighted pair connectivity')
            valid = edges[:, 0] != -1
            e = edges[valid]
            if (e[:, 2] < 0).any() or any((e[:, a] < 0).any() or (e[:, a] >= counts[a]).any()
                                         or not np.equal(e[:, a], e[:, a].astype(int)).all()
                                         for a in (0, 1)):
                raise ValueError('Invalid public connectivity')
        if pins.ndim != 2 or pins.shape[1] != 2:
            raise ValueError('Expected public pin XY coordinates')
        return cls(n, areas, b2b[b2b[:, 0] != -1], p2b[p2b[:, 0] != -1],
                   pins, constraints.astype(int), specs, math.sqrt(float(areas.sum())))

    @property
    def locked(self):
        return self.constraints[:, :2].any(1)

    @property
    def preplaced(self):
        return self.constraints[:, 1] != 0


def dimensions(p, log_ratios=None):
    """Exact free areas; compatible MIB groups share one ratio (or locked WH).

    Incompatible original-input MIB groups are explicitly left soft. Their area
    or locked specifications are never edited or dropped to obtain uniformity.
    """
    r = np.zeros(p.n) if log_ratios is None else np.asarray(log_ratios, float).copy()
    if r.shape != (p.n,) or not np.isfinite(r).all():
        raise ValueError('Malformed shape parameters')
    r = np.clip(r, -math.log(4), math.log(4))
    wh = np.column_stack((np.sqrt(p.areas) * np.exp(r / 2), np.sqrt(p.areas) * np.exp(-r / 2)))
    notes = []
    for group in sorted(set(p.constraints[:, 2]) - {0}):
        ids = np.flatnonzero(p.constraints[:, 2] == group)
        rigid = ids[p.locked[ids]]
        same_area = np.allclose(p.areas[ids], p.areas[ids[0]], rtol=1e-10, atol=1e-10)
        same_rigid = not len(rigid) or np.array_equal(
            p.specs[rigid, 2:], np.tile(p.specs[rigid[0], 2:], (len(rigid), 1)))
        if same_area and same_rigid and (not len(rigid) or
                np.max(np.abs(np.prod(p.specs[rigid[0], 2:]) / p.areas[ids] - 1)) < .009999):
            if len(rigid):
                wh[ids] = p.specs[rigid[0], 2:]
            else:
                ratio = float(r[ids].mean())
                wh[ids] = np.sqrt(p.areas[ids[0]]) * np.exp([ratio / 2, -ratio / 2])
        else:
            notes.append({'group': int(group), 'reason': 'public_area_or_locked_shape_conflict',
                          'members': ids.tolist(), 'hard_specs_preserved': True})
    wh[p.locked] = p.specs[p.locked, 2:]
    return wh, notes


def hard_check(p, pos):
    z = np.asarray(pos, float)
    if z.shape != (p.n, 4):
        return {'is_feasible': False, 'reason': 'wrong_output_shape'}
    reference = public_hard_reference(p.areas, p.constraints, p.specs)
    check = independent_hard_checks(z, p.areas, p.constraints, reference)
    independently_valid = check['is_feasible']
    check['negative_coordinate_blocks'] = np.flatnonzero((z[:, :2] < 0).any(1)).tolist()
    check['is_feasible'] = independently_valid and not check['negative_coordinate_blocks']
    official = official_evaluator()
    native = (official.check_overlap(z.tolist()) == 0
              and official.check_area_tolerance(z.tolist(), p.areas,
                                               skip_indices=set(np.flatnonzero(p.locked))) == 0
              and official.check_dimension_hard_constraints(z.tolist(), reference,
                                                           p.constraints, p.n) == 0)
    if native != independently_valid:
        raise AssertionError('Official/independent hard checks disagree')
    check['official_hard_valid'] = bool(native)
    return check


def legal_incumbent(p, wh):
    """Place free rectangles on a shelf beyond public anchors; no rescue loop.

    This intentionally simple safety incumbent is not a competitive baseline.
    Overlapping immutable anchors have no legal output; return an explicit error.
    """
    pos = np.column_stack((np.zeros((p.n, 2)), wh))
    pos[p.preplaced] = p.specs[p.preplaced]
    anchor = pos[p.preplaced]
    for i in range(len(anchor)):
        for j in range(i):
            overlap = np.minimum(anchor[i, :2] + anchor[i, 2:], anchor[j, :2] + anchor[j, 2:])
            overlap -= np.maximum(anchor[i, :2], anchor[j, :2])
            if (overlap > 1e-6).all():
                raise ValueError('Immutable public anchors overlap')
    start_y = float((anchor[:, 1] + anchor[:, 3]).max()) if len(anchor) else 0.
    shelf_width = max(p.scale, float(wh[:, 0].max()))
    x, y, row_h = 0., start_y, 0.
    for i in np.flatnonzero(~p.preplaced):
        w, h = wh[i]
        if x > 0 and x + w > shelf_width:
            x, y, row_h = 0., y + row_h, 0.
        pos[i, :2] = (x, y)
        x += w
        row_h = max(row_h, h)
    check = hard_check(p, pos)
    if not check['is_feasible']:
        raise RuntimeError('Safety incumbent failed hard checks: ' + str(check))
    return pos


def public_objective(p, pos):
    """Fixed input-normalized selection rule. True reference denominators absent."""
    z = np.asarray(pos, float)
    center = z[:, :2] + z[:, 2:] / 2
    hpwl, weight = 0., 0.
    for i, j, w in p.b2b:
        hpwl += w * np.abs(center[int(i)] - center[int(j)]).sum()
        weight += w
    for pin, i, w in p.p2b:
        hpwl += w * np.abs(p.pins[int(pin)] - center[int(i)]).sum()
        weight += w
    extent = (z[:, :2] + z[:, 2:]).max(0) - z[:, :2].min(0)
    bbox = float(extent.prod())
    from shapely.geometry import box
    from shapely.ops import unary_union
    groups = 0
    for g in set(p.constraints[:, 3]) - {0}:
        ids = np.flatnonzero(p.constraints[:, 3] == g)
        shape = unary_union([box(*z[i, :2], *(z[i, :2] + z[i, 2:])) for i in ids])
        groups += len(shape.geoms) - 1 if shape.geom_type == 'MultiPolygon' else 0
    mib = 0
    for g in set(p.constraints[:, 2]) - {0}:
        ids = np.flatnonzero(p.constraints[:, 2] == g)
        mib += len({tuple(np.round(z[i, 2:], 4)) for i in ids}) - 1
    lo, hi = z[:, :2].min(0), (z[:, :2] + z[:, 2:]).max(0)
    boundary = 0
    for i, code in enumerate(p.constraints[:, 4]):
        distances = {1: z[i, 0] - lo[0], 2: hi[0] - z[i, 0] - z[i, 2],
                     4: hi[1] - z[i, 1] - z[i, 3], 8: z[i, 1] - lo[1]}
        boundary += any(abs(value) >= 1e-6 for bit, value in distances.items() if code & bit)
    nsoft = (int((p.constraints[:, 4] != 0).sum())
             + sum(max(0, int((p.constraints[:, 3] == g).sum()) - 1)
                   for g in set(p.constraints[:, 3]) - {0})
             + sum(max(0, int((p.constraints[:, 2] == g).sum()) - 1)
                   for g in set(p.constraints[:, 2]) - {0}))
    value = .5 * hpwl / max(weight * p.scale, 1e-12) + .5 * bbox / p.areas.sum()
    value += 2 * (boundary + groups + mib) / max(nsoft, 1)
    return {'value': float(value), 'hpwl': float(hpwl), 'bbox_area': bbox,
            'boundary': int(boundary), 'grouping': int(groups), 'mib': int(mib),
            'uses_reference': False}
