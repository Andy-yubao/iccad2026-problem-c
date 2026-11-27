"""Independent E3 finite contact candidates; only legal solver inputs enter here."""
import math
import time
import numpy as np


def array(x):
    return np.asarray(x.detach().cpu() if hasattr(x, 'detach') else x, dtype=float)


class Problem:
    def __init__(self, n, areas, b2b, p2b, pins, constraints, specs):
        self.n = int(n)
        self.areas = array(areas)[:n]
        self.c = array(constraints)[:n, :5].astype(int)
        self.specs = np.full((n, 4), -1.)
        fixed = (self.c[:, 0] != 0) | (self.c[:, 1] != 0)
        pre = self.c[:, 1] != 0
        if np.any(fixed):
            supplied = array(specs)
            self.specs[fixed, 2:] = supplied[fixed, 2:]
            self.specs[pre, :2] = supplied[pre, :2]
        if np.any(self.areas <= 0) or np.any(~np.isfinite(self.areas)):
            raise ValueError('Nonpositive/nonfinite area')
        if np.any(self.specs[fixed, 2:] <= 0) or np.any(~np.isfinite(self.specs[fixed, 2:])):
            raise ValueError('Invalid immutable shape')
        if np.any(self.specs[pre, :2] < 0) or np.any(~np.isfinite(self.specs[pre, :2])):
            raise ValueError('Invalid anchor')
        self.scale = math.sqrt(float(self.areas.sum()))
        self.adj = np.zeros((n, n))
        self.pin_edges = [[] for _ in range(n)]
        for a, b, weight in array(b2b).reshape(-1, 3):
            if 0 <= a < n and 0 <= b < n and weight > 0:
                self.adj[int(a), int(b)] += weight
                self.adj[int(b), int(a)] += weight
        pins = array(pins)
        for a, b, weight in array(p2b).reshape(-1, 3):
            if 0 <= a < len(pins) and 0 <= b < n and weight > 0 and np.all(pins[int(a)] >= 0):
                self.pin_edges[int(b)].append((pins[int(a)], weight))
        self.order = sorted(np.flatnonzero(~pre).tolist(), key=lambda i: (-self.areas[i], i))
        self.initial = {int(i): self.specs[i].copy() for i in np.flatnonzero(pre)}
        if self.initial:
            r = np.asarray(list(self.initial.values()))
            for i in range(len(r)):
                if np.any(overlap(r[i:i+1], r[:i])):
                    raise ValueError('Overlapping immutable anchors')

    def shapes(self, i):
        if self.c[i, 0] or self.c[i, 1]:
            return self.specs[i:i+1, 2:].copy()
        # Search restriction only. All three retain the true original area.
        return np.asarray([[math.sqrt(self.areas[i] * r), math.sqrt(self.areas[i] / r)]
                           for r in (.5, 1., 2.)])


def overlap(candidates, placed):
    if not len(placed):
        return np.zeros((len(candidates), 0), dtype=bool)
    return np.all(np.minimum(candidates[:, None, :2] + candidates[:, None, 2:],
                             placed[None, :, :2] + placed[None, :, 2:]) -
                  np.maximum(candidates[:, None, :2], placed[None, :, :2]) > 1e-6, axis=2)


def candidate_pool(p, i, state):
    """Soft-blind pool and hard mask. Guaranteed right shelf plus contacts."""
    placed = np.asarray(list(state.values())).reshape(-1, 4)
    xmax = float(np.max(placed[:, 0] + placed[:, 2])) if len(placed) else 0.
    ymax = float(np.max(placed[:, 1] + placed[:, 3])) if len(placed) else 0.
    chunks = []
    for w, h in p.shapes(i):
        points = [(0., 0.), (xmax, 0.), (0., ymax)]
        for x, y, a, b in placed:
            points.extend(((x + a, y), (x + a, y + b - h), (x - w, y),
                           (x - w, y + b - h), (x, y + b), (x + a - w, y + b),
                           (x, y - h), (x + a - w, y - h)))
        xy = np.asarray(points)
        xy = xy[np.all(xy >= -1e-8, axis=1)].clip(0)
        chunks.append(np.column_stack((xy, np.full(len(xy), w), np.full(len(xy), h))))
    pool = np.unique(np.concatenate(chunks), axis=0)
    mask = ~np.any(overlap(pool, placed), axis=1)
    return pool, mask


def features(p, i, pool, state):
    ids = list(state)
    r = np.asarray(list(state.values())).reshape(-1, 4)
    xmax = np.max(r[:, 0] + r[:, 2]) if len(r) else 0.
    ymax = np.max(r[:, 1] + r[:, 3]) if len(r) else 0.
    right, top = pool[:, 0] + pool[:, 2], pool[:, 1] + pool[:, 3]
    W, H = np.maximum(xmax, right), np.maximum(ymax, top)
    center = pool[:, :2] + pool[:, 2:] / 2
    wire = np.zeros(len(pool))
    weight = 0.
    for j in ids:
        w = p.adj[i, j]
        wire += w * np.abs(center - (state[j][:2] + state[j][2:] / 2)).sum(axis=1)
        weight += w
    for pin, w in p.pin_edges[i]:
        wire += w * np.abs(center - pin).sum(axis=1)
        weight += w
    wire /= max(weight * p.scale, 1.)
    code = p.c[i, 4]
    boundary = np.zeros(len(pool))
    if code & 1:
        boundary += pool[:, 0] / p.scale
    if code & 2:
        boundary += (W - right) / p.scale
    if code & 4:
        boundary += (H - top) / p.scale
    if code & 8:
        boundary += pool[:, 1] / p.scale
    # Distance-to-contact is soft; it never removes any candidate.
    group = np.zeros(len(pool))
    members = [j for j in ids if p.c[i, 3] > 0 and p.c[j, 3] == p.c[i, 3]]
    if members:
        distances = []
        for j in members:
            x, y, w, h = state[j]
            dx = np.maximum(np.maximum(x - right, pool[:, 0] - x - w), 0)
            dy = np.maximum(np.maximum(y - top, pool[:, 1] - y - h), 0)
            distances.append((dx + dy) / p.scale)
        group = np.min(distances, axis=0)
    mib = np.zeros(len(pool))
    members = [j for j in ids if p.c[i, 2] > 0 and p.c[j, 2] == p.c[i, 2]]
    if members:
        mib = np.min([np.abs(np.log(pool[:, 2:] / state[j][2:])).sum(axis=1)
                      for j in members], axis=0)
    return np.column_stack((W * H / p.scale**2, wire, (W + H) / p.scale,
                            boundary, group, mib))


def heuristic_scores(f, soft=True):
    return f @ np.asarray([1., .3, .02, .3 if soft else 0., .3 if soft else 0., .1 if soft else 0.])


def boundary_violations(p, positions):
    r = np.asarray(positions)
    W, H = np.max(r[:, 0] + r[:, 2]), np.max(r[:, 1] + r[:, 3])
    count = 0
    for i, (x, y, w, h) in enumerate(r):
        c = p.c[i, 4]
        count += bool((c & 1 and abs(x) > 1e-5) or (c & 2 and abs(x+w-W) > 1e-5)
                      or (c & 4 and abs(y+h-H) > 1e-5) or (c & 8 and abs(y) > 1e-5))
    return count


def grouping_components(p, positions):
    """Positive-length edge contact graph; reports disconnected tight subgroups."""
    r = np.asarray(positions)
    result = {}
    for g in sorted(set(p.c[:, 3]) - {0}):
        todo = set(np.flatnonzero(p.c[:, 3] == g).tolist())
        components = 0
        while todo:
            stack = [todo.pop()]
            components += 1
            while stack:
                i = stack.pop()
                for j in list(todo):
                    a, b = r[i], r[j]
                    ox = min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])
                    oy = min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1])
                    if (abs(ox) < 1e-5 and oy > 1e-5) or (abs(oy) < 1e-5 and ox > 1e-5):
                        todo.remove(j)
                        stack.append(j)
        result[str(int(g))] = components
    return result


class Optimizer:
    def __init__(self, strategy='greedy', seconds=15, seed=0, scorer=None):
        if strategy not in ('greedy', 'no_soft', 'imitation'):
            raise ValueError(strategy)
        if strategy == 'imitation' and scorer is None:
            raise ValueError('imitation needs an explicitly loaded trained scorer')
        self.strategy, self.seconds, self.scorer = strategy, seconds, scorer

    def solve(self, *inputs):
        start = time.perf_counter()
        p = Problem(*inputs)
        state = dict(p.initial)
        raw = dict(p.initial)
        stats = dict(empty_candidate_count=0, fallback_count=0, geometry_solve_calls=0,
                     candidate_total=0, feasible_candidate_total=0, future_deadlock_count=0,
                     order=p.order, continuous_correction='identity; no coordinate optimizer',
                     strategy=self.strategy, deadline_shelf_count=0)
        for i in p.order:
            pool, mask = candidate_pool(p, i, state)
            f = features(p, i, pool, state)
            scores = heuristic_scores(f, self.strategy != 'no_soft')
            if self.scorer is not None:
                scores = -np.asarray(self.scorer(f)).reshape(-1)
            raw[i] = pool[int(np.argmin(scores))]
            ids = np.flatnonzero(mask)
            stats['candidate_total'] += len(pool)
            stats['feasible_candidate_total'] += len(ids)
            stats['geometry_solve_calls'] += 1
            if not len(ids):
                stats['empty_candidate_count'] += 1
                stats['fallback_count'] += 1
                raise RuntimeError('Right-shelf existence invariant violated')
            if time.perf_counter() - start > self.seconds - .1:
                # Every remaining block is still represented and hard checked.
                stats['deadline_shelf_count'] += 1
                stats['fallback_count'] += 1
                xmax = max((x+w for x,y,w,h in state.values()), default=0.)
                w,h = p.shapes(i)[0]
                state[i] = np.array([xmax, 0., w, h])
            else:
                state[i] = pool[ids[int(np.argmin(scores[ids]))]]
        positions = np.asarray([state[i] for i in range(p.n)])
        draft = np.asarray([raw[i] for i in range(p.n)])
        stats['solve_seconds'] = time.perf_counter() - start
        stats['repair_center_displacement_mean'] = float(np.mean(np.linalg.norm(
            positions[:, :2] + positions[:, 2:] / 2 - draft[:, :2] - draft[:, 2:] / 2, axis=1)))
        stats['group_contact_components'] = grouping_components(p, positions)
        stats['boundary_violations_global'] = boundary_violations(p, positions)
        self.last_stats = stats
        self.raw_positions = draft.tolist()
        return positions.tolist()
