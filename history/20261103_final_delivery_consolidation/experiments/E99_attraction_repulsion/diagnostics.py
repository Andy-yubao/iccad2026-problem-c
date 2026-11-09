"""Exact geometry reports only; never used as a force or acceptance objective.

Union sweep follows the local reporting approach recorded in E97 draft_metrics;
kept independent here, without importing another experiment's implementation.
"""
import numpy as np


def union_area(rectangles):
    r = np.asarray(rectangles, dtype=float).reshape(-1, 4)
    if not len(r):
        return 0.
    xs = np.unique(np.r_[r[:, 0], r[:, 0] + r[:, 2]])
    result = 0.
    for left, right in zip(xs[:-1], xs[1:]):
        active = r[(r[:, 0] < right) & (r[:, 0] + r[:, 2] > left)]
        if not len(active):
            continue
        intervals = sorted(zip(active[:, 1], active[:, 1] + active[:, 3]))
        a, b = intervals[0]
        height = 0.
        for c, d in intervals[1:]:
            if c > b:
                height += b - a
                a, b = c, d
            else:
                b = max(b, d)
        result += (right - left) * (height + b - a)
    return float(result)


def measures(positions):
    p = np.asarray(positions, dtype=float)
    low, high = p[:, :2], p[:, :2] + p[:, 2:]
    area = p[:, 2:].prod(1)
    ilow = np.maximum(low[:, None], low[None])
    size = np.maximum(0, np.minimum(high[:, None], high[None]) - ilow)
    inter = size.prod(-1)
    np.fill_diagonal(inter, 0)
    coverage = []
    for i in range(len(p)):
        ids = np.flatnonzero(inter[i] > 0)
        coverage.append(union_area(np.c_[ilow[i, ids], size[i, ids]]) / area[i])
    coverage = np.clip(coverage, 0, 1)
    bbox = float(np.prod(high.max(0) - low.min(0)))
    occupied = union_area(p)
    return dict(pair_intersection_over_area_sum=float(inter.sum() / 2 / area.sum()),
                overlap_pairs=int(np.count_nonzero(np.triu(inter, 1))),
                mean_module_coverage=float(coverage.mean()),
                modules_strictly_below_10pct=int((coverage < .1).sum()),
                fraction_strictly_below_10pct=float((coverage < .1).mean()),
                worst_module=int(np.argmax(coverage)), worst_module_coverage=float(coverage.max()),
                bbox_area=bbox, bbox_over_area_sum=float(bbox / area.sum()),
                exact_void_fraction=float((bbox - occupied) / bbox))
