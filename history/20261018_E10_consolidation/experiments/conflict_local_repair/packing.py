"""Input-only rectangular packing with immutable-obstacle support.

Copied from baseline packing.py on 2026-10-16; only input preparation retained.  Only masked specifications supplied to
``solve`` are accepted: free-block label positions are never needed.  NumPy is
the only dependency.  The output uses (x, y, width, height), in input order.

This is a feasibility-first constructive heuristic, not an optimal packer.
Movable blocks are placed around every already placed rectangle, including
floating preplaced obstacles.  Candidate failures always have a legal shelf
fallback.  Inconsistent immutable input specifications raise ValueError.
"""

from dataclasses import dataclass
import math
import time
from typing import Optional, Sequence

import numpy as np


def _numpy(value, dtype=float):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


@dataclass
class PackingProblem:
    n: int
    areas: np.ndarray
    constraints: np.ndarray
    specs: np.ndarray
    b2b: np.ndarray
    p2b: np.ndarray
    pins: np.ndarray
    fixed: np.ndarray
    preplaced: np.ndarray
    adjacency: np.ndarray
    pin_weight: np.ndarray
    pin_sum: np.ndarray
    anchors: np.ndarray
    total_area: float
    scale: float
    total_wire_weight: float


def prepare_problem(block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints,
                    target_positions=None):
    """Validate legal input and cache connectivity for repeated pack calls.

    Padding edges/pins and rows beyond block_count are discarded.  A fixed
    block requires positive specified dimensions; a preplaced block additionally
    requires finite nonnegative coordinates.  Only the fields allowed by each
    constraint flag are copied into ``specs`` even if a caller passes more.
    """
    n = int(block_count)
    if n < 0:
        raise ValueError("block_count must be nonnegative")
    areas = _numpy(area_targets).reshape(-1)[:n].copy()
    if len(areas) != n or np.any(~np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("Every block needs a finite positive area target")
    raw_constraints = _numpy(constraints)
    c = np.zeros((n, 5), dtype=np.int64)
    if raw_constraints.size:
        if raw_constraints.ndim != 2 or raw_constraints.shape[0] < n:
            raise ValueError("constraints must have one row per block")
        cols = min(raw_constraints.shape[1], 5)
        c[:, :cols] = raw_constraints[:n, :cols].astype(np.int64)
    fixed, preplaced = c[:, 0] != 0, c[:, 1] != 0
    specs = np.full((n, 4), -1.0, dtype=float)
    if np.any(fixed | preplaced):
        if target_positions is None:
            raise ValueError("Fixed and preplaced blocks require target_positions")
        supplied = _numpy(target_positions)
        if supplied.ndim != 2 or supplied.shape[0] < n or supplied.shape[1] < 4:
            raise ValueError("target_positions must have shape [n,4]")
        specs[fixed | preplaced, 2:] = supplied[fixed | preplaced, 2:4]
        specs[preplaced, :2] = supplied[preplaced, :2]
        sizes = specs[fixed | preplaced, 2:]
        if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
            raise ValueError("Immutable dimensions must be finite and positive")
        coords = specs[preplaced, :2]
        if np.any(~np.isfinite(coords)) or np.any(coords < 0):
            raise ValueError("Immutable coordinates must be finite and nonnegative")
        locked = specs[preplaced]
        for k, rect in enumerate(locked):
            others = locked[:k]
            if len(others) and np.any(_overlaps(rect[None, :], others)):
                raise ValueError("Preplaced rectangles overlap; no legal packing exists")

    pins = _numpy(pins_pos).reshape(-1, 2).copy()

    def edges(value, limits):
        result = _numpy(value).reshape(-1, 3).copy()
        if not len(result):
            return result
        good = np.all(np.isfinite(result), axis=1) & (result[:, 2] > 0)
        good &= (result[:, 0] >= 0) & (result[:, 0] < limits[0])
        good &= (result[:, 1] >= 0) & (result[:, 1] < limits[1])
        good &= (result[:, 0] == np.floor(result[:, 0]))
        good &= (result[:, 1] == np.floor(result[:, 1]))
        result = result[good]
        return result

    b2b = edges(b2b_connectivity, (n, n))
    p2b = edges(p2b_connectivity, (len(pins), n))
    if len(p2b):
        pin_indices = p2b[:, 0].astype(int)
        p2b = p2b[np.all(np.isfinite(pins[pin_indices]), axis=1)
                   & np.all(pins[pin_indices] >= 0, axis=1)]
    adjacency = np.zeros((n, n), dtype=float)
    if len(b2b):
        i, j = b2b[:, 0].astype(int), b2b[:, 1].astype(int)
        np.add.at(adjacency, (i, j), b2b[:, 2])
        np.add.at(adjacency, (j, i), b2b[:, 2])
    np.fill_diagonal(adjacency, 0)
    pin_weight, pin_sum = np.zeros(n), np.zeros((n, 2))
    if len(p2b):
        p, b = p2b[:, 0].astype(int), p2b[:, 1].astype(int)
        np.add.at(pin_weight, b, p2b[:, 2])
        np.add.at(pin_sum, b, p2b[:, 2, None] * pins[p])
    total_area = float(areas[~(fixed | preplaced)].sum()
                       + np.prod(specs[fixed | preplaced, 2:], axis=1).sum())
    scale = math.sqrt(max(total_area, 1e-30))
    anchors = np.full((n, 2), 0.5 * scale)
    if pin_weight.sum() > 0:
        anchors[:] = pin_sum.sum(axis=0) / pin_weight.sum()
    if np.any(preplaced):
        anchors[preplaced] = specs[preplaced, :2] + specs[preplaced, 2:] / 2
    degree = adjacency.sum(axis=1) + pin_weight
    # Damped Jacobi provides a cheap net-driven reference without label geometry.
    for _ in range(32):
        update = (adjacency @ anchors + pin_sum) / np.maximum(degree[:, None], 1e-30)
        update[degree == 0] = anchors[degree == 0]
        anchors[~preplaced] = 0.2 * anchors[~preplaced] + 0.8 * update[~preplaced]
    total_wire_weight = float(b2b[:, 2].sum() + p2b[:, 2].sum())
    return PackingProblem(n, areas, c, specs, b2b, p2b, pins, fixed,
                          preplaced, adjacency, pin_weight, pin_sum, anchors,
                          total_area, scale, total_wire_weight)


def _overlaps(candidates, rectangles):
    """Strict positive-area intersection, with no permissive overlap epsilon."""
    return ((candidates[:, None, 0] < rectangles[None, :, 0] + rectangles[None, :, 2])
            & (candidates[:, None, 0] + candidates[:, None, 2] > rectangles[None, :, 0])
            & (candidates[:, None, 1] < rectangles[None, :, 1] + rectangles[None, :, 3])
            & (candidates[:, None, 1] + candidates[:, None, 3] > rectangles[None, :, 1]))

