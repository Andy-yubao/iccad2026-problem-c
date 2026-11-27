"""Input-only rectangular packing with immutable-obstacle support.

The public interface is ``prepare_problem(...)`` followed by ``pack(problem,
order, ratios, target_width=...)``.  Only masked specifications supplied to
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


def default_order(problem, use_soft=True):
    """Deterministic group-contiguous, constraint-first ordering."""
    p = problem
    movable = np.flatnonzero(~p.preplaced).tolist()
    importance = p.adjacency.sum(axis=1) + p.pin_weight
    groups = p.constraints[:, 3] if use_soft else np.zeros(p.n)
    priorities = []
    for i in movable:
        boundary = int(p.constraints[i, 4]) if use_soft else 0
        # A group shares its largest member's size priority to stay contiguous.
        members = np.flatnonzero(groups == groups[i]) if groups[i] > 0 else np.array([i])
        priorities.append((0 if boundary else 1,
                           -float(p.areas[members].sum()),
                           int(groups[i]) if groups[i] > 0 else p.n + i,
                           -float(importance[i]), -float(p.areas[i]), i))
    return np.array([row[-1] for row in sorted(priorities)], dtype=int)


def _dimensions(p, ratios, use_soft):
    if ratios is None:
        ratios = np.ones(p.n)
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim == 0:
        ratios = np.full(p.n, float(ratios))
    if ratios.shape != (p.n,) or np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("ratios must be finite positive w/h values, one per block")
    widths = np.sqrt(p.areas * ratios)
    heights = p.areas / widths
    immutable = p.fixed | p.preplaced
    widths[immutable], heights[immutable] = p.specs[immutable, 2], p.specs[immutable, 3]
    if use_soft:
        for group in np.unique(p.constraints[:, 2]):
            if group <= 0:
                continue
            members = np.flatnonzero(p.constraints[:, 2] == group)
            rigid = members[immutable[members]]
            if len(rigid):
                representative = rigid[0]
                w, h = widths[representative], heights[representative]
                for i in members[~immutable[members]]:
                    # The official area constraint permits one percent error.
                    # Shared immutable dimensions are legal within that interval;
                    # keep a tiny numerical margin rather than landing on 1%.
                    if abs(w * h - p.areas[i]) < p.areas[i] * 0.009999999:
                        widths[i], heights[i] = w, h
            else:
                ratio = math.exp(float(np.log(ratios[members]).mean()))
                lower = float((p.areas[members] * 0.990000001).max())
                upper = float((p.areas[members] * 1.009999999).min())
                if lower <= upper:
                    common_area = float(np.clip(math.exp(float(np.log(p.areas[members]).mean())),
                                                lower, upper))
                    w = math.sqrt(common_area * ratio)
                    widths[members], heights[members] = w, common_area / w
                else:
                    widths[members] = np.sqrt(p.areas[members] * ratio)
                    heights[members] = p.areas[members] / widths[members]
    return widths, heights


def _candidate_points(rectangles, w, h, width, desired):
    """Bottom/left compaction at obstacle-edge events and the net anchor."""
    if not len(rectangles):
        return np.array([[0., 0., w, h],
                         [max(0., min(width - w, desired[0] - w / 2)), 0., w, h]])
    left, bottom = rectangles[:, 0], rectangles[:, 1]
    right, top = left + rectangles[:, 2], bottom + rectangles[:, 3]
    xs = np.unique(np.r_[0., width - w, left, right, left - w, right - w,
                         desired[0] - w / 2])
    xs = xs[(xs >= 0) & (xs + w <= width)]
    ys = np.zeros(len(xs))
    crosses_x = (xs[:, None] < right) & (xs[:, None] + w > left)
    for _ in range(len(rectangles) + 1):
        collision = crosses_x & (ys[:, None] < top) & (ys[:, None] + h > bottom)
        if not np.any(collision):
            break
        ys = np.maximum(ys, np.where(collision, top, 0).max(axis=1))
    first = np.column_stack((xs, ys, np.full(len(xs), w), np.full(len(xs), h)))
    ys = np.unique(np.r_[0., bottom, top, bottom - h, top - h, desired[1] - h / 2])
    ys = ys[ys >= 0]
    xs = np.zeros(len(ys))
    crosses_y = (ys[:, None] < top) & (ys[:, None] + h > bottom)
    for _ in range(len(rectangles) + 1):
        collision = crosses_y & (xs[:, None] < right) & (xs[:, None] + w > left)
        if not np.any(collision):
            break
        xs = np.maximum(xs, np.where(collision, right, 0).max(axis=1))
    inside = xs + w <= width
    second = np.column_stack((xs[inside], ys[inside], np.full(inside.sum(), w),
                              np.full(inside.sum(), h)))
    return np.concatenate((first, second), axis=0)


def pack(problem, order: Optional[Sequence[int]] = None, ratios=None,
         target_width=None, *, wire_weight=0.7, boundary_weight=2.0,
         group_weight=2.0, shape_trials=3, mode="bottom_left", use_soft=True,
         deadline=None, proposal_centers=None):
    """Construct a nonnegative, nonoverlapping layout from legal specifications.

    ``ratios`` gives width/height per block.  Fixed and preplaced dimensions
    ignore it. ``target_width`` is a strip-width proposal, expanded only when a
    rigid rectangle, preplaced extent, or top-boundary shelf requires it.
    ``wire_weight`` multiplies incremental weighted Manhattan wirelength divided
    by total edge weight and sqrt(total area); geometric costs use area units.

    Group contact and boundary penalties are soft.  MIB-compatible groups share
    shape; free top-boundary blocks are placed on one common ceiling. ``mode``
    may be ``bottom_left``, ``shelf`` (stronger compact rows), or ``net`` (extra
    attraction to the input-only harmonic net anchors). ``deadline`` is an
    absolute time.monotonic() value: near expiration shape exploration stops,
    and after expiration a cheap legal shelf completes the remaining blocks.
    """
    p = problem
    if mode not in ("bottom_left", "shelf", "net"):
        raise ValueError("Unknown packing mode")
    if p.n == 0:
        return np.empty((0, 4), dtype=float)
    widths, heights = _dimensions(p, ratios, use_soft)
    width = float(target_width) if target_width is not None else p.scale
    if not math.isfinite(width) or width <= 0:
        raise ValueError("target_width must be finite and positive")
    width = max(width, float(widths.max()))
    if np.any(p.preplaced):
        width = max(width, float((p.specs[p.preplaced, 0] + widths[p.preplaced]).max()))
    if order is None:
        order = default_order(p, use_soft)
    else:
        order = np.asarray(order, dtype=int).reshape(-1)
        if len(set(order.tolist())) != len(order) or np.any((order < 0) | (order >= p.n)):
            raise ValueError("order must contain distinct valid block indices")
        provided = set(order.tolist())
        missing = np.array([i for i in range(p.n) if i not in provided], dtype=int)
        order = np.concatenate((order, missing))
        order = order[~p.preplaced[order]]
    boundary = p.constraints[:, 4] if use_soft else np.zeros(p.n, dtype=int)
    top_ids = [int(i) for i in order if boundary[i] & 4 and not boundary[i] & 8]
    # Top shelf width is decided before placing any right-boundary rectangle.
    # Thus a later ceiling cannot invalidate already satisfied right boundaries.
    if top_ids:
        roof_width = math.fsum(float(widths[i]) for i in top_ids)
        # A small ulp margin covers different left/right accumulation orders
        # without creating tiny overlaps where the two shelf cursors meet.
        for _ in top_ids:
            roof_width = float(np.nextafter(roof_width, math.inf))
        width = max(width, roof_width)
    normal = [int(i) for i in order if i not in set(top_ids)]
    positions = np.zeros((p.n, 4), dtype=float)
    placed = p.preplaced.copy()
    positions[p.preplaced] = p.specs[p.preplaced]
    current_right = float((positions[placed, 0] + positions[placed, 2]).max()) if placed.any() else 0.
    current_top = float((positions[placed, 1] + positions[placed, 3]).max()) if placed.any() else 0.
    wire_norm = max(p.total_wire_weight * p.scale, 1e-30)
    penalty_norm = max(p.n, 1)
    immutable = p.fixed | p.preplaced

    for i in normal:
        rectangles = positions[placed]
        indices = np.flatnonzero(placed)
        w0, h0 = widths[i], heights[i]
        expired = deadline is not None and time.monotonic() >= deadline
        if expired:
            # Placing above all immutable/movable rectangles is always legal.
            # This is a completion path, not a partially filled answer.
            x = width - w0 if boundary[i] & 2 else 0.
            positions[i] = (x, current_top, w0, h0)
            placed[i] = True
            current_right = max(current_right, x + w0)
            current_top += h0
            continue
        neighbours = p.adjacency[i, indices]
        neighbour_weight = float(neighbours.sum())
        total_local_weight = neighbour_weight + p.pin_weight[i]
        desired = p.anchors[i].copy()
        if total_local_weight > 0:
            centers = rectangles[:, :2] + rectangles[:, 2:] / 2
            desired = (neighbours @ centers + p.pin_sum[i]) / total_local_weight
        if proposal_centers is not None:
            desired = np.asarray(proposal_centers[i], dtype=float)
        shape_count = max(1, int(shape_trials))
        if immutable[i] or (use_soft and p.constraints[i, 2] > 0):
            shape_count = 1
        scales = [1.0]
        if shape_count > 1:
            # Explore aspect ratio multiplicatively while preserving exact area.
            scales += [math.exp(v) for v in np.linspace(-0.4, 0.4, shape_count)
                       if abs(v) > 1e-12]
            scales = scales[:shape_count]
        all_candidates = []
        for factor in scales:
            if deadline is not None and time.monotonic() >= deadline and all_candidates:
                break
            w = min(width, w0 * factor) if not immutable[i] else w0
            h = p.areas[i] / w if not immutable[i] else h0
            # Preserve exact matching dimensions of an equal-area MIB/fixed pair.
            if factor == 1 and shape_count == 1:
                w, h = w0, h0
            candidates = _candidate_points(rectangles, w, h, width, desired)
            # Explicit boundary anchors include the right edge, which cannot be
            # obtained by a pure left compaction in an empty strip.
            extra = [[0., current_top, w, h], [width - w, current_top, w, h]]
            if not len(rectangles):
                extra += [[width - w, 0., w, h]]
            candidates = np.r_[candidates, extra]
            if len(rectangles):
                candidates = candidates[~np.any(_overlaps(candidates, rectangles), axis=1)]
            candidates = candidates[(candidates[:, :2] >= 0).all(axis=1)]
            all_candidates.append(candidates)
        candidates = np.concatenate(all_candidates, axis=0)
        centers = candidates[:, :2] + candidates[:, 2:] / 2
        right = candidates[:, 0] + candidates[:, 2]
        top = candidates[:, 1] + candidates[:, 3]
        new_right, new_top = np.maximum(current_right, right), np.maximum(current_top, top)
        score = new_right * new_top / max(p.total_area, 1e-30)
        if proposal_centers is not None:
            score += 0.1 * np.abs(centers - desired).sum(axis=1) / p.scale
        if neighbour_weight:
            neighbour_centers = rectangles[:, :2] + rectangles[:, 2:] / 2
            score += float(wire_weight) * (np.abs(centers[:, None, :] - neighbour_centers)
                                          .sum(axis=2) @ neighbours) / wire_norm
        if p.pin_weight[i]:
            pin_edges = p.p2b[p.p2b[:, 1] == i]
            coords = p.pins[pin_edges[:, 0].astype(int)]
            score += float(wire_weight) * (np.abs(centers[:, None, :] - coords)
                                          .sum(axis=2) @ pin_edges[:, 2]) / wire_norm
        if mode == "net":
            score += 0.05 * np.abs(centers - p.anchors[i]).sum(axis=1) / (p.scale * penalty_norm)
        elif mode == "shelf":
            score += 0.1 * candidates[:, 1] / (p.scale * penalty_norm)
        if boundary[i]:
            satisfied = np.ones(len(candidates), dtype=bool)
            if boundary[i] & 1:
                satisfied &= candidates[:, 0] == 0
            if boundary[i] & 2:
                satisfied &= right == width
            if boundary[i] & 8:
                satisfied &= candidates[:, 1] == 0
            if boundary[i] & 4:
                satisfied &= top == new_top
            score += float(boundary_weight) * (~satisfied) / penalty_norm
        group = p.constraints[i, 3] if use_soft else 0
        if group > 0:
            same = rectangles[p.constraints[indices, 3] == group]
            if len(same):
                sx, sy = same[:, 0], same[:, 1]
                sr, st = sx + same[:, 2], sy + same[:, 3]
                vertical = ((candidates[:, None, 0] == sr) | (right[:, None] == sx))
                vertical &= ((candidates[:, None, 1] < st) & (top[:, None] > sy))
                horizontal = ((candidates[:, None, 1] == st) | (top[:, None] == sy))
                horizontal &= ((candidates[:, None, 0] < sr) & (right[:, None] > sx))
                contact = np.any(vertical | horizontal, axis=1)
                score += float(group_weight) * (~contact) / penalty_norm
        # Tiny tie-breaks select the lower/left placement without changing a
        # material geometric/wirelength trade-off.
        score += 1e-10 * (candidates[:, 1] + candidates[:, 0] * 0.01) / p.scale
        best = int(np.argmin(score))
        positions[i] = candidates[best]
        placed[i] = True
        current_right, current_top = max(current_right, right[best]), max(current_top, top[best])

    if top_ids:
        # A common ceiling produces exact top-edge equality; side corners are
        # allocated first and all remaining top blocks fill the middle shelf.
        ceiling = current_top + max(float(heights[i]) for i in top_ids)
        left_ids = [i for i in top_ids if boundary[i] & 1]
        right_ids = [i for i in top_ids if boundary[i] & 2 and i not in left_ids]
        middle_ids = [i for i in top_ids if i not in left_ids and i not in right_ids]
        cursor = 0.
        for i in left_ids + middle_ids:
            w, h = widths[i], heights[i]
            positions[i] = (cursor, max(current_top, ceiling - h), w, h)
            cursor += w
        cursor = width
        for i in right_ids:
            w, h = widths[i], heights[i]
            cursor -= w
            positions[i] = (cursor, max(current_top, ceiling - h), w, h)
    return positions


__all__ = ["PackingProblem", "prepare_problem", "default_order", "pack"]
