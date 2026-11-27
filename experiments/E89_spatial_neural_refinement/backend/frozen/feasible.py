"""E1 input-only quality proxy and legality. Copied from baseline 2026-10-16."""
import math
import numpy as np

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
