"""Exact DAG transitive reduction of positive-size separation constraints."""
import heapq


def reduce_edges(edges, n):
    """Keep cyclic axes unchanged; a path implies a direct separation edge.

    For hard inequalities x_j >= x_i + width_i, every positive-size path
    is at least as strong as the direct edge. In an elastic shape LP the
    hard-feasible region is preserved, but the slack objective is changed.
    """
    result = []
    for axis in (0, 1):
        part = [(int(i), int(j), a) for i, j, a in edges if a == axis]
        adjacent = [set() for _ in range(n)]
        degree = [0] * n
        for i, j, _ in part:
            if j not in adjacent[i]:
                adjacent[i].add(j)
                degree[j] += 1
        queue = [i for i, d in enumerate(degree) if d == 0]
        heapq.heapify(queue)
        order = []
        while queue:
            i = heapq.heappop(queue)
            order.append(i)
            for j in sorted(adjacent[i]):
                degree[j] -= 1
                if degree[j] == 0:
                    heapq.heappush(queue, j)
        if len(order) != n:
            result.extend(part)
            continue
        rank = {i: k for k, i in enumerate(order)}
        reach = [0] * n
        for i in reversed(order):
            covered = 0
            for j in sorted(adjacent[i], key=rank.get):
                if not covered & (1 << j):
                    result.append((i, j, axis))
                covered |= (1 << j) | reach[j]
            reach[i] = covered
    return result
