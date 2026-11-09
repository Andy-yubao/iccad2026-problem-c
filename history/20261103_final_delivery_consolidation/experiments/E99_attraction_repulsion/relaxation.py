"""Finite-range central forces; translations only, no task/score objectives.

This is an overdamped geometric heuristic, not the gradient of a claimed
rectangle potential. In particular the directional contact radius depends on
angle. It neither certifies nonoverlap nor promises monotonic overlap reduction.
"""
from dataclasses import asdict, dataclass
from numbers import Integral
import time

import numpy as np


@dataclass(frozen=True)
class Config:
    iterations: int = 4
    step_size: float = 0.5
    step_fraction: float = 0.02
    total_fraction: float = 0.06
    attraction_range: float = 0.5
    attraction_strength: float = 0.25
    repulsion_strength: float = 1.0
    mass_weighted: bool = True

    def __post_init__(self):
        if not isinstance(self.mass_weighted, bool):
            raise ValueError('mass_weighted must be boolean')
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, Integral) or not 0 <= self.iterations <= 8:
            raise ValueError('iterations must be an integer in [0, 8]')
        limits = {'step_size': 1., 'step_fraction': .02, 'total_fraction': .06,
                  'attraction_range': 1., 'attraction_strength': 1., 'repulsion_strength': 1.}
        for name, upper in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f'{name} must be finite and in (0, {upper}]')


def _positions(value):
    p = np.array(value, dtype=np.float64, copy=True)
    if p.ndim != 2 or p.shape[1] != 4 or not len(p):
        raise ValueError('positions must be nonempty N x 4 XYWH')
    if not np.isfinite(p).all() or (p[:, 2:] <= 0).any() or (p[:, :2] < 0).any():
        raise ValueError('finite positive dimensions and nonnegative XY required')
    # Reject overflow before constructing any pair geometry.
    with np.errstate(over='ignore', invalid='ignore'):
        high = p[:, :2] + p[:, 2:]
        size = np.hypot(p[:, 2], p[:, 3])
    if not np.isfinite(high).all() or not np.isfinite(size).all():
        raise ValueError('geometry exceeds float64 range')
    return p


def _pairs(p, config):
    """Unique pairs, unit direction i->j, signed radial gap, scalar force."""
    i, j = np.triu_indices(len(p), 1)
    center = p[:, :2] + p[:, 2:] / 2
    delta = center[j] - center[i]
    distance = np.hypot(delta[:, 0], delta[:, 1])
    unit = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
    tied = distance == 0
    # Stable ID-based symmetry breaking only where no center line exists.
    # Several directions prevent coincident clusters from sharing one axis.
    angles = (((i[tied] + 1) * 73856093 ^ (j[tied] + 1) * 19349663) % 104729) * (2 * np.pi / 104729)
    unit[tied] = np.column_stack((np.cos(angles), np.sin(angles)))
    extent = p[i, 2:] / 2 + p[j, 2:] / 2
    radius_axis = np.divide(extent, np.abs(unit), out=np.full_like(extent, np.inf), where=np.abs(unit) > 0)
    contact = radius_axis.min(axis=1)
    gap = distance - contact
    tolerance = 32 * np.finfo(np.float64).eps * np.maximum(distance, contact)
    gap[np.abs(gap) <= tolerance] = 0
    local = np.minimum(p[i, 2:].min(axis=1), p[j, 2:].min(axis=1))
    cutoff = config.attraction_range * local
    t = np.clip(gap / cutoff, 0, 1)
    # Compact-support attraction is zero both at contact and at the cutoff.
    attractive = config.attraction_strength * np.maximum(gap, 0) * (1 - t) ** 2
    repulsive = -config.repulsion_strength * local * np.tanh(np.maximum(-gap, 0) / local)
    scalar = attractive + repulsive
    return i, j, unit, gap, scalar


def _cap(vector, radius):
    length = np.hypot(vector[:, 0], vector[:, 1])
    factor = np.minimum(1., np.divide(radius, length, out=np.ones_like(length), where=length > 0))
    return vector * factor[:, None]


def relax(positions, preplaced=None, *, config=None, return_history=False):
    """Return (new XYWH, diagnostics), leaving the input untouched.

    Per-step and total path-length budgets are fractions of EACH module's
    initial shortest side. Total path also bounds displacement from the input.
    Fixed-shape blocks may move; preplaced blocks must be explicitly masked.
    XY>=0 is projected solely as a coordinate-domain rule; no bounding frame,
    group, MIB, wire, port, label, or scoring inputs enter this operator.
    """
    start = time.perf_counter()
    config = Config() if config is None else config
    if not isinstance(config, Config):
        raise TypeError('config must be Config')
    p = _positions(positions)
    original = p.copy()
    n = len(p)
    if preplaced is None:
        locked = np.zeros(n, dtype=bool)
    else:
        values = np.asarray(preplaced)
        if values.shape != (n,) or not np.isin(values, [0, 1]).all():
            raise ValueError('preplaced must be N boolean/0/1 values')
        locked = values.astype(bool)
    local = p[:, 2:].min(axis=1)
    if config.mass_weighted:
        with np.errstate(over='ignore', under='ignore'):
            area = p[:, 2] * p[:, 3]
        if not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError('module area must be finite and positive')
        # The common median only sets units: mass ratios are exactly area ratios.
        mass = area / np.median(area)
        if not np.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError('normalized module mass is outside float64 range')
    else:
        mass = np.ones(n)
    path = np.zeros(n)
    trace = []
    history = [p.copy()] if return_history else None
    stop = 'iteration_limit'
    for iteration in range(config.iterations):
        i, j, unit, gap, scalar = _pairs(p, config)
        force = scalar[:, None] * unit
        net = np.zeros((n, 2))
        np.add.at(net, i, force)
        np.add.at(net, j, -force)
        # One scalar damping factor for the entire layout retains equal and
        # opposite pair forces; high-degree overlaps cannot amplify an update.
        active = scalar != 0
        degree = np.bincount(np.r_[i[active], j[active]], minlength=n)
        damping = max(1, int(degree.max()))
        step = config.step_size * net / damping / mass[:, None]
        step[locked] = 0
        remaining = np.maximum(0, config.total_fraction * local - path)
        radius = np.minimum(config.step_fraction * local, remaining)
        if config.mass_weighted:
            # A common safety scale retains the inverse-mass response. Clipping
            # each particle separately could make two saturated large/small
            # blocks move according to their side lengths instead of their mass.
            lengths_before_cap = np.hypot(step[:, 0], step[:, 1])
            ratios = np.divide(radius, lengths_before_cap, out=np.ones(n), where=lengths_before_cap > 0)
            step *= min(1., float(ratios.min()))
        else:
            # Explicit frozen v1 response, retained only as the no-mass control.
            step = _cap(step, radius)
        proposed = np.maximum(0, p[:, :2] + step)
        proposed[locked] = original[locked, :2]
        actual = proposed - p[:, :2]
        lengths = np.hypot(actual[:, 0], actual[:, 1])
        if not np.isfinite(proposed).all():
            raise FloatingPointError('nonfinite proposed geometry')
        path += lengths
        p[:, :2] = proposed
        trace.append(dict(iteration=iteration + 1, attractive_pairs=int((scalar > 0).sum()),
                          repulsive_pairs=int((scalar < 0).sum()), damping=damping,
                          max_step_fraction=float((lengths / local).max()),
                          max_path_fraction=float((path / local).max())))
        if return_history:
            history.append(p.copy())
        if not lengths.any():
            stop = 'no_movement'
            break
    displacement = np.hypot(*(p[:, :2] - original[:, :2]).T)
    # Scaled floating-point tolerance is only for auditing arithmetic roundoff.
    eps = 128 * np.finfo(np.float64).eps * np.maximum(1, np.abs(original[:, :2]).max(axis=1))
    if (path > config.total_fraction * local + eps).any() or (displacement > config.total_fraction * local + eps).any():
        raise AssertionError('movement budget exceeded')
    if not np.array_equal(p[:, 2:], original[:, 2:]) or not np.array_equal(p[locked], original[locked]):
        raise AssertionError('immutable geometry changed')
    diagnostics = dict(config=asdict(config),iterations=len(trace),stop_reason=stop,
                       mass_model='area_over_median_area' if config.mass_weighted else 'unit_mass_v1',
                       min_normalized_mass=float(mass.min()),max_normalized_mass=float(mass.max()),
                       max_displacement_fraction=float((displacement / local).max()),
                       mean_displacement_fraction=float((displacement / local).mean()),
                       max_path_fraction=float((path / local).max()),trace=trace,
                       seconds=time.perf_counter() - start)
    if return_history:
        diagnostics['history'] = np.stack(history)
    return p, diagnostics
