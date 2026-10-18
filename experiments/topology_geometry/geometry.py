"""Fixed input-only shapes; separation preferences -> exact difference constraints.

Directions for i<j are i-left-j, j-left-i, i-below-j, j-below-i.
All experiments use the same least-coordinate objective. This is a constraint
graph solver for fixed dimensions, not a convexification of variable shapes.
"""
import time
import numpy as np
from .input_geometry import prepare_problem, _overlaps


def fixed_shapes(p):
    dimensions = np.repeat(np.sqrt(p.areas)[:, None], 2, axis=1)
    locked = p.fixed | p.preplaced
    dimensions[locked] = p.specs[locked, 2:]
    return dimensions


def proposal(p, dimensions):
    out = np.column_stack((np.maximum(0., p.anchors - dimensions / 2), dimensions))
    out[p.preplaced] = p.specs[p.preplaced]
    return out


class DifferenceGraph:
    """Max-plus all-pairs closure. Origin n has coordinate zero.

    An edge u->v with weight w means x[v]>=x[u]+w. The closure
    detects positive cycles, including paths conflicting with fixed anchors.
    Distances from origin give the componentwise least feasible coordinates,
    minimizing sum(coordinates) among placements with these relations.
    """
    def __init__(self, sizes, anchors, fixed):
        self.n = len(sizes)
        self.sizes = np.asarray(sizes)
        self.distance = np.full((self.n + 1, self.n + 1), -np.inf)
        np.fill_diagonal(self.distance, 0.)
        self.edges = []
        self.reach = np.eye(self.n, dtype=bool)
        for i in range(self.n):
            assert self.insert(self.n, i, 0., record=False)
        for i in np.flatnonzero(fixed):
            if not self.insert(self.n, i, float(anchors[i]), record=False):
                raise ValueError("Inconsistent anchor coordinate")
            if not self.insert(i, self.n, -float(anchors[i]), record=False):
                raise ValueError("Inconsistent anchor coordinate")

    def insert(self, u, v, weight=None, record=True):
        w = float(self.sizes[u]) if weight is None else float(weight)
        # Every geometric edge has positive size, so an ordinary graph cycle
        # is impossible even before taking anchored distances into account.
        if record and self.reach[v, u]:
            return False
        d = self.distance
        if d[v, u] + w > 1e-10:
            return False
        candidate = d[:, u, None] + w + d[None, v, :]
        np.maximum(d, candidate, out=d)
        if record:
            self.edges.append((int(u), int(v)))
            self.reach |= self.reach[:, u, None] & self.reach[None, v, :]
        return True

    def coordinates(self):
        return self.distance[self.n, :self.n].copy()


def shelf(p, dimensions):
    """Guaranteed completion around immutable anchors; same dimensions always."""
    out = np.zeros((p.n, 4), dtype=float)
    out[:, 2:] = dimensions
    out[p.preplaced] = p.specs[p.preplaced]
    top = float(np.max(out[p.preplaced, 1] + out[p.preplaced, 3])) if p.preplaced.any() else 0.
    cursor, row_height = 0., 0.
    width = max(p.scale, float(dimensions[:, 0].max(initial=0)))
    margin = 1e-8 * max(p.scale, 1.)
    for i in np.flatnonzero(~p.preplaced):
        w, h = dimensions[i]
        if cursor + w > width and cursor > 0:
            cursor, top, row_height = 0., top + row_height + margin, 0.
        out[i, :2] = cursor, top
        cursor += w + margin
        row_height = max(row_height, h)
    return out


def hard_valid(p, out):
    if out.shape != (p.n, 4) or not np.isfinite(out).all() or (out[:, :2] < 0).any():
        return False
    locked = p.fixed | p.preplaced
    if not np.array_equal(out[locked, 2:], p.specs[locked, 2:]):
        return False
    if not np.array_equal(out[p.preplaced], p.specs[p.preplaced]):
        return False
    if np.any(np.abs(np.prod(out[~locked, 2:], axis=1) - p.areas[~locked]) > p.areas[~locked] * 1e-6):
        return False
    overlap = _overlaps(out, out)
    return not np.triu(overlap, 1).any()


def direction_scores(p, dims, strategy, rng):
    pairs = np.column_stack(np.triu_indices(p.n, 1))
    if strategy == "random":
        return pairs, rng.standard_normal((len(pairs), 4))
    centers = proposal(p, dims)[:, :2] + dims / 2
    i, j = pairs.T
    delta = (centers[j] - centers[i]) / np.maximum((dims[i] + dims[j]) / 2, 1e-12)
    # Hash-free deterministic spread resolves equal net-driven centers without
    # exposing label positions or dimensions. This is part of the heuristic.
    near = np.linalg.norm(delta, axis=1) < 1e-8
    delta[near, 0] = ((j[near] % 8) - (i[near] % 8)) / 8
    delta[near, 1] = ((j[near] // 8) - (i[near] // 8)) / 8
    return pairs, np.column_stack((delta[:, 0], -delta[:, 0], delta[:, 1], -delta[:, 1]))


def solve_relations(p, dims, pairs, scores, seconds=15., start=None):
    started = time.perf_counter() if start is None else start
    deadline = started + seconds
    raw = proposal(p, dims)
    graphs = [DifferenceGraph(dims[:, a], p.specs[:, a], p.preplaced) for a in (0, 1)]
    stats = {"pairs": len(pairs), "cycle_rejections": 0, "anchor_conflict_rejections": 0,
             "modified_relation_count": 0, "geometric_solve_count": 1,
             "fallback": False, "fallback_reason": None, "soft_boundary_forced": False,
             "objective": "componentwise least nonnegative positions for fixed directions/shapes"}
    preferred = np.argsort(-scores, axis=1, kind="stable")
    confidence = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    for k in np.argsort(-confidence, kind="stable"):
        if time.perf_counter() > deadline - .05:
            stats.update(fallback=True, fallback_reason="relation_budget")
            break
        i, j = map(int, pairs[k])
        # Exact touching is sufficient; a numerical margin on movable pairs
        # protects the final all-pair check. Two immutable anchors keep specs.
        margin = 1e-8 * max(p.scale, 1.) if not (p.preplaced[i] and p.preplaced[j]) else 0.
        accepted = False
        for rank, direction in enumerate(preferred[k]):
            axis = int(direction) // 2
            u, v = (i, j) if direction % 2 == 0 else (j, i)
            graph = graphs[axis]
            if graph.reach[v, u]:
                stats["cycle_rejections"] += 1
                continue
            if not graph.insert(u, v, dims[u, axis] + margin):
                stats["anchor_conflict_rejections"] += 1
                continue
            stats["modified_relation_count"] += int(rank != 0)
            accepted = True
            break
        if not accepted:
            stats.update(fallback=True, fallback_reason="no_consistent_direction")
            break
    corrected = np.column_stack((graphs[0].coordinates(), graphs[1].coordinates(), dims))
    corrected[p.preplaced] = p.specs[p.preplaced]
    final = corrected.copy()
    if stats["fallback"] or not hard_valid(p, final):
        if not stats["fallback"]:
            stats.update(fallback=True, fallback_reason="final_geometry_check")
        final = shelf(p, dims)
    stats["modified_relation_fraction"] = stats["modified_relation_count"] / max(len(pairs), 1)
    stats["geometry_seconds"] = time.perf_counter() - started
    stats["correction_mean_displacement"] = float(np.linalg.norm(corrected[:, :2] - raw[:, :2], axis=1).mean())
    stats["fallback_mean_displacement"] = float(np.linalg.norm(final[:, :2] - corrected[:, :2], axis=1).mean())
    assert hard_valid(p, final), "Final geometry failed input-only hard checks"
    return raw, corrected, final, stats


class Optimizer:
    def __init__(self, strategy="heuristic", seconds=15., seed=0, checkpoint=None):
        self.strategy, self.seconds, self.seed, self.checkpoint = strategy, seconds, seed, checkpoint

    def solve(self, *inputs):
        started = time.perf_counter()
        p = prepare_problem(*inputs)
        dims = fixed_shapes(p)
        if self.strategy == "learned":
            if self.checkpoint is None:
                raise ValueError("learned requires an explicitly trained checkpoint")
            from .model import predict
            pairs, scores = predict(p, dims, self.checkpoint)
        else:
            if self.strategy not in ("heuristic", "random"):
                raise ValueError("Unknown relation strategy")
            pairs, scores = direction_scores(p, dims, self.strategy, np.random.default_rng(self.seed))
        raw, corrected, final, self.last_stats = solve_relations(p, dims, pairs, scores, self.seconds, started)
        self.stages = {"raw": raw, "corrected": corrected, "final": final}
        return final
