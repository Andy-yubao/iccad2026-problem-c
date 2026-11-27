"""Native kernels that keep the frozen solver's order and comparisons."""

import numpy as np
from numba import njit

from .repair.backend.frozen.feasible import (
    Objective,
)


@njit(cache=True)
def reduce_edges(edges, n):
    """Transitive reduction with smallest-index Kahn order and cyclic fallback."""
    out = np.empty_like(edges)
    count = 0
    words = (n + 63) // 64
    for axis in range(2):
        adjacent = np.zeros((n, n), np.bool_)
        indegree = np.zeros(n, np.int64)
        for i, j, edge_axis in edges:
            if axis == edge_axis and not adjacent[i, j]:
                adjacent[i, j] = True
                indegree[j] += 1
        order = np.empty(n, np.int64)
        size = 0
        for position in range(n):
            first = -1
            for i in range(n):
                if indegree[i] == 0:
                    first = i
                    break
            if first == -1:
                break
            order[position] = first
            size += 1
            indegree[first] = -1
            for j in range(n):
                if adjacent[first, j]:
                    indegree[j] -= 1
        if size != n:
            for k in range(len(edges)):
                if edges[k, 2] == axis:
                    out[count] = edges[k]
                    count += 1
            continue
        reachable = np.zeros((n, words), np.uint64)
        for position in range(n - 1, -1, -1):
            i = order[position]
            for next_position in range(position + 1, n):
                j = order[next_position]
                if adjacent[i, j]:
                    word = j // 64
                    bit = np.uint64(1) << np.uint64(j % 64)
                    if not reachable[i, word] & bit:
                        out[count, 0] = i
                        out[count, 1] = j
                        out[count, 2] = axis
                        count += 1
                    reachable[i, word] |= bit
                    for w in range(words):
                        reachable[i, w] |= reachable[j, w]
    return out[:count]


@njit(cache=True)
def _hard_valid(layout, areas, fixed, preplaced, specs):
    upper = layout[:, :2] + layout[:, 2:]
    for i in range(len(layout)):
        x, y, w, h = layout[i]
        if not (
            np.isfinite(x) and np.isfinite(y) and np.isfinite(w) and np.isfinite(h)
        ):
            return False
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
        if fixed[i] or preplaced[i]:
            if abs(w - specs[i, 2]) > 1e-8 or abs(h - specs[i, 3]) > 1e-8:
                return False
        elif abs(w * h - areas[i]) / areas[i] > 0.01:
            return False
        if preplaced[i] and (
            abs(x - specs[i, 0]) > 1e-8 or abs(y - specs[i, 1]) > 1e-8
        ):
            return False
        for j in range(i):
            overlap_x = min(upper[i, 0], upper[j, 0]) - max(x, layout[j, 0])
            overlap_y = min(upper[i, 1], upper[j, 1]) - max(y, layout[j, 1])
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                return False
    return True


def hard_valid(layout, problem):
    layout = np.asarray(layout, dtype=np.float64)
    return layout.shape == (problem.n, 4) and _hard_valid(
        layout, problem.areas, problem.fixed, problem.preplaced, problem.specs
    )


@njit(cache=True)
def _violations(layout, codes, members, offsets):
    upper = layout[:, :2] + layout[:, 2:]
    x_lo = layout[:, 0].min()
    y_lo = layout[:, 1].min()
    x_hi = upper[:, 0].max()
    y_hi = upper[:, 1].max()
    boundary = 0
    for i, code in enumerate(codes):
        if (
            (code & 1 and not abs(layout[i, 0] - x_lo) < 1e-6)
            or (code & 2 and not abs(upper[i, 0] - x_hi) < 1e-6)
            or (code & 4 and not abs(upper[i, 1] - y_hi) < 1e-6)
            or (code & 8 and not abs(layout[i, 1] - y_lo) < 1e-6)
        ):
            boundary += 1
    grouping = 0
    roots = np.arange(len(members))
    for g in range(len(offsets) - 1):
        first, end = offsets[g], offsets[g + 1]
        grouping += end - first - 1
        for a in range(first, end):
            i = members[a]
            for b in range(first, a):
                j = members[b]
                dx = min(upper[i, 0], upper[j, 0]) - max(layout[i, 0], layout[j, 0])
                dy = min(upper[i, 1], upper[j, 1]) - max(layout[i, 1], layout[j, 1])
                if (dx >= 0 and dy > 0) or (dx > 0 and dy >= 0):
                    ra, rb = a, b
                    while roots[ra] != ra:
                        ra = roots[ra]
                    while roots[rb] != rb:
                        rb = roots[rb]
                    if ra != rb:
                        roots[rb] = ra
                        grouping -= 1
    return boundary, grouping, 0


class NativeObjective(Objective):
    def __init__(self, problem):
        super().__init__(problem)
        self.members = (
            np.concatenate(self.groups) if self.groups else np.empty(0, dtype=np.int64)
        )
        self.offsets = np.r_[
            0, np.cumsum([len(g) for g in self.groups], dtype=np.int64)
        ]

    def violations(self, layout):
        if self.mibs:
            return super().violations(layout)
        return _violations(layout, self.codes, self.members, self.offsets)
