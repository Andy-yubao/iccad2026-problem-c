"""E5 input-only objective and reinsertion. Copied from baseline 2026-10-16."""
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
