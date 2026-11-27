"""Frozen SOCP construction with vectorized singleton area normalization.

The no-MIB path retains the same matrix values, row order, cone order,
Clarabel settings and numerical tolerances as E165 fast_geometry.step.
"""

import time

import clarabel
import numpy as np
from scipy.sparse import coo_matrix

from experiments.E165_joint_l0.evaluation.repair.fast_geometry import Rows, contact_tree


def step(p, seed, edges, full_edges, nsoft, *, boundary, contact, elastic, reweighted):
    start = time.perf_counter()
    n, s = p.n, p.scale
    z = seed / s
    wh = z[:, 2:]
    locked = p.fixed | p.preplaced
    free = np.flatnonzero(~locked)
    ns = nsoft
    tree = np.asarray(contact_tree(seed, p, full_edges), dtype=int).reshape(-1, 3)
    edges = np.asarray(edges, dtype=int).reshape(-1, 3)
    e = len(edges)
    t = len(tree)
    xy = np.arange(2 * n).reshape(n, 2)
    dims = 2 * n + np.arange(2 * n).reshape(n, 2)
    lo = 4 * n + np.arange(2)
    hi = 4 * n + 2 + np.arange(2)
    move = 4 * n + 4 + np.arange(2 * n).reshape(n, 2)
    requested = np.flatnonzero(p.constraints[:, 4])
    boundary_ids = 6 * n + 4 + np.arange(len(requested))
    elastic_ids = 6 * n + 4 + len(requested) + np.arange(e if elastic else 0)
    contact_ids = 6 * n + 4 + len(requested) + len(elastic_ids) + np.arange(t)
    nv = 6 * n + 4 + len(requested) + len(elastic_ids) + t
    costs = np.zeros(nv)
    box_weight = np.ones(2)
    costs[lo] = -box_weight
    costs[hi] = box_weight
    costs[move] = 0.005 / n
    if elastic:
        costs[elastic_ids] = elastic / n
    eq = Rows()
    ineq = Rows()
    eq.add(dims[locked].ravel(), 1, wh[locked].ravel())
    eq.add(xy[p.preplaced].ravel(), 1, z[p.preplaced, :2].ravel())
    ineq.add(np.arange(nv), -1, 0.0)
    ineq.add(dims[free].ravel(), -1, -0.4 * wh[free].ravel())
    ineq.add(dims[free].ravel(), 1, 2.5 * wh[free].ravel())
    ineq.add(
        dims[free],
        wh[free, ::-1],
        2 * np.sqrt(wh[free].prod(1) * (p.areas[free] / s**2) * 1.009999),
    )
    axis = np.tile(np.arange(2), n)
    ineq.add(np.column_stack((lo[axis], xy.ravel())), [1, -1], 0)
    ineq.add(np.column_stack((xy.ravel(), dims.ravel(), hi[axis])), [1, 1, -1], 0)
    ineq.add(np.column_stack((xy.ravel(), move.ravel())), [1, -1], z[:, :2].ravel())
    ineq.add(np.column_stack((xy.ravel(), move.ravel())), [-1, -1], -z[:, :2].ravel())
    if len(requested):
        bblo = z[:, :2].min(0)
        bbhi = (z[:, :2] + wh).max(0)
        distances = np.column_stack(
            (
                z[:, 0] - bblo[0],
                bbhi[0] - z[:, 0] - wh[:, 0],
                bbhi[1] - z[:, 1] - wh[:, 1],
                z[:, 1] - bblo[1],
            )
        )
        bits = np.array([1, 2, 4, 8])
        mask = (p.constraints[requested, 4, None].astype(int) & bits) != 0
        prior = np.where(mask, distances[requested], -np.inf).max(1)
        costs[boundary_ids] = (
            2 / ns / (np.maximum(prior, 0) + 0.02 * np.sqrt(wh[requested].prod(1)))
            if reweighted
            else boundary / ns
        )
        for column, (a, upper) in enumerate(
            [(0, False), (0, True), (1, True), (1, False)]
        ):
            use = mask[:, column]
            ids = requested[use]
            slack = boundary_ids[use]
            if upper:
                ineq.add(
                    np.column_stack(
                        (np.full(len(ids), hi[a]), xy[ids, a], dims[ids, a], slack)
                    ),
                    [1, -1, -1, -1],
                    0,
                )
            else:
                ineq.add(
                    np.column_stack((xy[ids, a], np.full(len(ids), lo[a]), slack)),
                    [1, -1, -1],
                    0,
                )
    edge_row_start = ineq.count
    if e:
        i, j, a = edges.T
        columns = np.column_stack((xy[i, a], dims[i, a], xy[j, a]))
        if elastic:
            ineq.add(np.column_stack((columns, elastic_ids)), [1, 1, -1, -1], 0)
        else:
            ineq.add(columns, [1, 1, -1], 0)
    if t:
        i, j, a = tree.T
        b = 1 - a
        gap = np.maximum.reduce(
            (
                np.zeros(t),
                z[j, a] - z[i, a] - wh[i, a],
                z[i, b] - z[j, b] - wh[j, b],
                z[j, b] - z[i, b] - wh[i, b],
            )
        )
        costs[contact_ids] = (
            2
            / ns
            / (gap + 0.02 * np.minimum(np.sqrt(wh[i].prod(1)), np.sqrt(wh[j].prod(1))))
            if reweighted
            else contact / ns
        )
        shared = 0.02 * np.minimum(wh[i, b], wh[j, b])
        ineq.add(
            np.column_stack((xy[j, a], xy[i, a], dims[i, a], contact_ids)),
            [1, -1, -1, -1],
            0,
        )
        ineq.add(
            np.column_stack((xy[i, b], xy[j, b], dims[j, b], contact_ids)),
            [1, -1, -1, -1],
            -shared,
        )
        ineq.add(
            np.column_stack((xy[j, b], xy[i, b], dims[i, b], contact_ids)),
            [1, -1, -1, -1],
            -shared,
        )
    # IRLS diagonal majorization, aggregated per block before sparse assembly.
    centers = z[:, :2] + wh / 2
    alpha = np.zeros((n, 2))
    beta = np.zeros((n, 2))
    norm = max(p.total_wire_weight, 1e-12)
    i, j = np.nonzero(np.triu(p.adjacency, 1))
    w = p.adjacency[i, j] / norm
    if len(i):
        prior = centers[i] - centers[j]
        local = w[:, None] / np.sqrt(prior**2 + 0.02**2)
        mid = (centers[i] + centers[j]) / 2
        np.add.at(alpha, i, local)
        np.add.at(alpha, j, local)
        np.add.at(beta, i, local * mid)
        np.add.at(beta, j, local * mid)
    if len(p.p2b):
        pin, i, w = p.p2b.T
        pin = pin.astype(int)
        i = i.astype(int)
        target = p.pins[pin, :2] / s
        local = (
            0.5 * (w / norm)[:, None] / np.sqrt((centers[i] - target) ** 2 + 0.02**2)
        )
        np.add.at(alpha, i, local)
        np.add.at(beta, i, local * target)
    costs[xy] -= 2 * beta
    costs[dims] -= beta
    pi = np.r_[xy.ravel(), dims.ravel(), xy.ravel()]
    pj = np.r_[xy.ravel(), dims.ravel(), dims.ravel()]
    pv = np.r_[(2 * alpha).ravel(), (0.5 * alpha).ravel(), alpha.ravel()]
    hessian = coo_matrix((pv, (pi, pj)), shape=(nv, nv)).tocsc()
    ii = []
    jj = []
    vv = []
    rhs = []
    offset = 0
    for block in (eq, ineq):
        if block.count:
            ii.append(np.concatenate(block.i) + offset)
            jj.extend(block.j)
            vv.extend(block.v)
            rhs.extend(block.b)
        offset += block.count
    cones = [clarabel.ZeroConeT(eq.count), clarabel.NonnegativeConeT(ineq.count)]
    if len(free):
        rr = offset + 3 * np.arange(len(free))
        ii.append(np.column_stack((rr, rr, rr + 1, rr + 1)).ravel())
        jj.append(
            np.column_stack(
                (dims[free, 0], dims[free, 1], dims[free, 0], dims[free, 1])
            ).ravel()
        )
        vv.append(np.tile([-1, -1, -1, 1], len(free)))
        cone_rhs = np.column_stack(
            (np.zeros((len(free), 2)), 2 * np.sqrt(p.areas[free] / s**2 * 0.990001))
        ).ravel()
        rhs.append(cone_rhs)
        cones.extend(clarabel.SecondOrderConeT(3) for _ in free)
        offset += 3 * len(free)
    matrix = coo_matrix(
        (np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))),
        shape=(offset, nv),
    ).tocsc()
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 100
    settings.tol_feas = 1e-10
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    assembled = time.perf_counter()
    solver = clarabel.DefaultSolver(
        hessian, costs, matrix, np.concatenate(rhs), cones, settings
    )
    initialized = time.perf_counter()
    result = solver.solve()
    stats = {
        "status": str(result.status),
        "native_seconds": result.solve_time,
        "seconds": time.perf_counter() - start,
        "assembly_seconds": assembled - start,
        "setup_seconds": initialized - assembled,
        "iterations": result.iterations,
        "variables": nv,
        "constraints": offset,
    }
    if str(result.status) == "PrimalInfeasible" and not elastic and e:
        ray = np.asarray(result.z)[
            eq.count + edge_row_start : eq.count + edge_row_start + e
        ]
        ranked = np.argsort(-np.abs(ray))[:8]
        stats["conflicting_edges"] = [
            [int(edges[k, 0]), int(edges[k, 1]), int(edges[k, 2]), float(abs(ray[k]))]
            for k in ranked
        ]
    if (
        str(result.status) not in ("Solved", "AlmostSolved", "MaxIterations")
        or result.r_prim > 2e-6
    ):
        return None, stats
    values = np.asarray(result.x)
    out = np.column_stack((values[xy], values[dims])) * s
    out[locked, 2:] = p.specs[locked, 2:]
    out[p.preplaced] = p.specs[p.preplaced]
    # Without MIB every shape class is a singleton. Apply the same clamp
    # and square-root correction to the whole free-block array at once.
    actual = out[free, 2:].prod(1)
    lower = p.areas[free] * 0.9900001 / actual
    upper = p.areas[free] * 1.009999 / actual
    factor = np.where(lower > upper, 1.0, np.clip(1.0, lower, upper))
    out[free, 2:] *= np.sqrt(factor)[:, None]
    return out, stats
