"""Input-only obstacle-aware construction and feasible coordinate search.

No labels or reference metrics enter this module.  The objective normalizes by
input area and net weight; it is a search surrogate, not the official score.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.baseline.packing import prepare_problem, pack


class Objective:
    def __init__(self, problem, use_soft=True):
        self.p = problem
        self.use_soft = use_soft
        self.groups = [np.flatnonzero(problem.constraints[:, 3] == g)
                       for g in np.unique(problem.constraints[:, 3]) if g > 0]
        self.mibs = [np.flatnonzero(problem.constraints[:, 2] == g)
                     for g in np.unique(problem.constraints[:, 2]) if g > 0]
        self.codes = problem.constraints[:, 4].astype(int)
        self.nsoft = max(1, int(np.count_nonzero(self.codes)) +
                         sum(len(g) - 1 for g in self.groups + self.mibs))
        effective = problem.areas.copy()
        locked = problem.fixed | problem.preplaced
        effective[locked] = np.prod(problem.specs[locked, 2:], axis=1)
        self.area_scale = max(float(effective.sum()), 1e-12)
        weights = problem.b2b[:, 2].sum() + problem.p2b[:, 2].sum()
        self.wire_scale = max(float(weights) * math.sqrt(self.area_scale) / 2, 1e-12)
        self.bi = problem.b2b[:, :2].astype(int)
        self.pi = problem.p2b[:, :2].astype(int)

    def violations(self, pos):
        lo = pos[:, :2].min(axis=0)
        hi = (pos[:, :2] + pos[:, 2:]).max(axis=0)
        touching = np.column_stack((np.abs(pos[:, 0] - lo[0]) < 1e-6,
                                    np.abs(pos[:, 0] + pos[:, 2] - hi[0]) < 1e-6,
                                    np.abs(pos[:, 1] + pos[:, 3] - hi[1]) < 1e-6,
                                    np.abs(pos[:, 1] - lo[1]) < 1e-6))
        required = (self.codes[:, None] & np.array([1, 2, 4, 8])) != 0
        boundary = int(np.any(required & ~touching, axis=1).sum())
        mib = sum(len({(round(float(pos[i, 2]), 4), round(float(pos[i, 3]), 4))
                       for i in g}) - 1 for g in self.mibs)
        grouping = 0
        for g in self.groups:
            r = pos[g]
            lower, upper = r[:, :2], r[:, :2] + r[:, 2:]
            overlap = np.minimum(upper[:, None], upper[None, :]) - np.maximum(lower[:, None], lower[None, :])
            # Exact contact, including ordinary roundoff overlap; a positive
            # gap is never treated as a shared edge. Point contact is excluded.
            touch = ((overlap[:, :, 0] >= 0) & (overlap[:, :, 1] > 0)) | \
                    ((overlap[:, :, 0] > 0) & (overlap[:, :, 1] >= 0))
            unseen = set(range(len(g)))
            components = 0
            while unseen:
                todo = [unseen.pop()]
                components += 1
                while todo:
                    neighbors = unseen.intersection(np.flatnonzero(touch[todo.pop()]).tolist())
                    unseen.difference_update(neighbors)
                    todo.extend(neighbors)
            grouping += components - 1
        return boundary, grouping, mib

    def __call__(self, pos):
        p = self.p
        centers = pos[:, :2] + pos[:, 2:] / 2
        wire = 0.0
        if len(self.bi):
            wire += float((np.abs(centers[self.bi[:, 0]] - centers[self.bi[:, 1]]).sum(axis=1) * p.b2b[:, 2]).sum())
        if len(self.pi):
            wire += float((np.abs(p.pins[self.pi[:, 0]] - centers[self.pi[:, 1]]).sum(axis=1) * p.p2b[:, 2]).sum())
        area = float(np.prod((pos[:, :2] + pos[:, 2:]).max(axis=0) - pos[:, :2].min(axis=0)))
        quality = .5 + .5 * (area / self.area_scale + wire / self.wire_scale)
        soft = sum(self.violations(pos)) if self.use_soft else 0
        return math.log(max(quality, 1e-12)) + 2 * soft / self.nsoft


def is_legal(pos, p):
    if pos.shape != (p.n, 4) or not np.isfinite(pos).all() or np.any(pos[:, 2:] <= 0) or np.any(pos[:, :2] < -1e-9):
        return False
    locked = p.fixed | p.preplaced
    if np.any(np.abs(pos[locked, 2:] - p.specs[locked, 2:]) > 1e-8):
        return False
    if np.any(np.abs(pos[p.preplaced] - p.specs[p.preplaced]) > 1e-8):
        return False
    free = ~locked
    if np.any(np.abs(np.prod(pos[free, 2:], axis=1) - p.areas[free]) > .01 * p.areas[free] + 1e-10):
        return False
    overlap = np.minimum((pos[:, :2] + pos[:, 2:])[:, None], (pos[:, :2] + pos[:, 2:])[None]) - \
              np.maximum(pos[:, None, :2], pos[None, :, :2])
    bad = (overlap[:, :, 0] > 1e-8) & (overlap[:, :, 1] > 1e-8)
    np.fill_diagonal(bad, False)
    return not bad.any()


def row_control(p):
    pos = np.zeros((p.n, 4), dtype=np.float64)
    pos[p.preplaced] = p.specs[p.preplaced]
    cursor = max(0.0, float((pos[:, 0] + pos[:, 2]).max()))
    for i in np.flatnonzero(~p.preplaced):
        wh = p.specs[i, 2:] if p.fixed[i] else np.repeat(math.sqrt(p.areas[i]), 2)
        pos[i] = [cursor, 0, *wh]
        cursor += wh[0]
    return pos


def weighted_median(values, weights, default):
    if len(values) == 0 or weights.sum() <= 0:
        return default
    order = np.argsort(values)
    return values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]]


def reinsertion_candidates(pos, i, wh, p, rng, use_soft):
    """Feasible face-aligned moves, ranked by local wire/area/contact proxy."""
    w, h = wh
    others = np.delete(pos, i, axis=0)
    high = (pos[:, :2] + pos[:, 2:]).max(axis=0)
    low = pos[:, :2].min(axis=0)
    centers = pos[:, :2] + pos[:, 2:] / 2
    weights = p.adjacency[i]
    peers = np.flatnonzero(weights > 0)
    pin_edges = p.p2b[p.p2b[:, 1] == i]
    net_xy = np.concatenate((centers[peers], p.pins[pin_edges[:, 0].astype(int)]))
    net_w = np.concatenate((weights[peers], pin_edges[:, 2]))
    aim = np.array([weighted_median(net_xy[:, d], net_w, centers[i, d]) for d in range(2)]) - np.array([w, h]) / 2
    # Face positions include both ends of each edge and the wire median.
    x, y, ow, oh = others.T
    rows = [pos[i, :2][None], aim[None]]
    for xx in (x - w, x + ow):
        rows.extend((np.column_stack((xx, y)), np.column_stack((xx, y + oh - h)),
                     np.column_stack((xx, np.clip(np.full(len(x), aim[1]), y - h / 2, y + oh - h / 2)))))
    for yy in (y - h, y + oh):
        rows.extend((np.column_stack((x, yy)), np.column_stack((x + ow - w, yy)),
                     np.column_stack((np.clip(np.full(len(x), aim[0]), x - w / 2, x + ow - w / 2), yy))))
    # Edge-constrained relocations may need a position far from the median.
    xs = np.unique(np.r_[low[0], high[0] - w, aim[0], x, x + ow - w])
    ys = np.unique(np.r_[low[1], high[1] - h, aim[1], y, y + oh - h])
    rows.extend((np.column_stack((np.full(len(ys), low[0]), ys)),
                 np.column_stack((np.full(len(ys), high[0] - w), ys)),
                 np.column_stack((xs, np.full(len(xs), low[1]))),
                 np.column_stack((xs, np.full(len(xs), high[1] - h)))))
    xy = np.unique(np.concatenate(rows), axis=0)
    xy = xy[np.all(xy >= 0, axis=1)]
    if not len(xy):
        return np.empty((0, 4))
    ext = xy + [w, h]
    ov = np.minimum(ext[:, None], others[None, :, :2] + others[None, :, 2:]) - np.maximum(xy[:, None], others[None, :, :2])
    valid = ~np.any((ov[:, :, 0] > 1e-9) & (ov[:, :, 1] > 1e-9), axis=1)
    xy = xy[valid]
    if not len(xy):
        return np.empty((0, 4))
    cent = xy + [w / 2, h / 2]
    wl = (np.abs(cent[:, None] - net_xy[None]).sum(axis=2) * net_w).sum(axis=1)
    area = np.prod(np.maximum(high, xy + [w, h]) - np.minimum(low, xy), axis=1)
    proxy = wl / max(float(net_w.sum()) * math.sqrt(p.areas.sum()), 1e-9) + area / p.areas.sum()
    chosen = set(np.argsort(proxy)[:5].tolist())
    if use_soft:
        code = int(p.constraints[i, 4])
        if code:
            touches = np.column_stack((np.abs(xy[:, 0] - low[0]) < 1e-6,
                                       np.abs(xy[:, 0] + w - high[0]) < 1e-6,
                                       np.abs(xy[:, 1] + h - high[1]) < 1e-6,
                                       np.abs(xy[:, 1] - low[1]) < 1e-6))
            sat = np.all(touches | ((code & np.array([1, 2, 4, 8])) == 0), axis=1)
            ids = np.flatnonzero(sat)
            chosen.update(ids[np.argsort(proxy[ids])[:4]].tolist())
        gid = p.constraints[i, 3]
        if gid:
            group = pos[(p.constraints[:, 3] == gid) & (np.arange(p.n) != i)]
            if len(group):
                ovg = np.minimum((xy + [w, h])[:, None], group[None, :, :2] + group[None, :, 2:]) - np.maximum(xy[:, None], group[None, :, :2])
                contact = np.any(((ovg[:, :, 0] >= 0) & (ovg[:, :, 1] > 0)) | ((ovg[:, :, 0] > 0) & (ovg[:, :, 1] >= 0)), axis=1)
                ids = np.flatnonzero(contact)
                chosen.update(ids[np.argsort(proxy[ids])[:5]].tolist())
    if len(xy) > 2:
        chosen.update(rng.choice(len(xy), min(2, len(xy)), replace=False).tolist())
    return np.column_stack((xy[sorted(chosen)], np.tile(wh, (len(chosen), 1))))


class Optimizer:
    def __init__(self, strategy="search", seconds=10.0, seed=0):
        if strategy not in ("row", "pack", "search", "no_soft"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy, self.seconds, self.seed = strategy, float(seconds), int(seed)
        self.last_stats = {}

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        started = time.monotonic()
        deadline = started + max(.05, self.seconds)
        p = prepare_problem(block_count, area_targets, b2b_connectivity, p2b_connectivity,
                            pins_pos, constraints, target_positions)
        best = row_control(p)
        if self.strategy == "row":
            return best.tolist()
        soft = self.strategy != "no_soft"
        objective = Objective(p, soft)
        best_score = objective(best)
        rng = np.random.default_rng(self.seed)
        effective = p.areas.copy()
        locked = p.fixed | p.preplaced
        effective[locked] = np.prod(p.specs[locked, 2:], axis=1)
        scale = math.sqrt(effective.sum())
        made, iterations, accepted = 0, 0, 0
        # A small fixed portfolio explores outline shape, not SA temperature.
        factors = [1.0, 1.3, .77, 1.7, .59]
        construction_end = deadline if self.strategy == "pack" else started + max(.05, self.seconds * .35)
        for k, factor in enumerate(factors):
            if time.monotonic() >= construction_end:
                break
            order = None
            if k >= 3:
                order = rng.permutation(p.n).tolist()
            candidate = pack(p, order=order, target_width=scale * factor,
                             wire_weight=.35, boundary_weight=.8, group_weight=.8,
                             shape_trials=3, use_soft=soft, deadline=construction_end)
            made += 1
            if is_legal(candidate, p):
                score = objective(candidate)
                if score < best_score:
                    best, best_score = candidate.copy(), score
        current, current_score = best.copy(), best_score
        movable = np.flatnonzero(~p.preplaced)
        while self.strategy != "pack" and len(movable) and time.monotonic() < deadline:
            i = int(rng.choice(movable))
            wh = current[i, 2:].copy()
            # Fixed dimensions never rotate. MIB shapes stay tied; unconstrained
            # soft blocks may change aspect ratio at exactly their input area.
            if not locked[i] and not (soft and p.constraints[i, 2]) and rng.random() < .25:
                size = math.sqrt(p.areas[i])
                wh[0] = np.clip(wh[0] * math.exp(rng.normal(0, .22)), size / 6, size * 6)
                wh[1] = p.areas[i] / wh[0]
            candidates = reinsertion_candidates(current, i, wh, p, rng, soft)
            trials = []
            old = current[i].copy()
            for rectangle in candidates:
                if time.monotonic() >= deadline:
                    break
                current[i] = rectangle
                trials.append((objective(current), rectangle.copy()))
            current[i] = old
            iterations += 1
            if not trials:
                continue
            score, rectangle = min(trials, key=lambda t: t[0])
            remain = max(0, (deadline - time.monotonic()) / max(self.seconds, .01))
            temperature = .012 * remain ** 2 + .00005
            if score < current_score or rng.random() < math.exp(min(0.0, (current_score - score) / temperature)):
                current[i] = rectangle
                current_score = score
                accepted += 1
                if score < best_score:
                    best, best_score = current.copy(), score
            if iterations % 80 == 0:
                current, current_score = best.copy(), best_score
        if not is_legal(best, p):
            best = row_control(p)
        self.last_stats = {"constructed": made, "iterations": iterations, "accepted": accepted,
                           "surrogate_log_cost": objective(best), "search_seconds": time.monotonic() - started,
                           "soft_counts": list(objective.violations(best))}
        return best.tolist()


MyOptimizer = Optimizer


if __name__ == "__main__":
    # Same JSON schema as the official executable wrapper; stdout stays clean.
    request = json.load(sys.stdin)
    solver = Optimizer()
    positions = solver.solve(**request)
    json.dump({"positions": positions}, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
