"""Ordered E98 search using native graph reduction and exact soft checks.

Traversal, acceptance order, Clarabel settings and work budgets follow the
frozen E165 solver. Keeping this entry separate preserves that run identity.
"""

import time
import numpy as np

from experiments.E165_joint_l0.evaluation.solver import BUDGET, seed_pack
from experiments.E165_joint_l0.evaluation.repair.backend.frozen.packing import (
    _dimensions,
)
from experiments.E165_joint_l0.evaluation.repair.backend.frozen.feasible import (
    row_control,
)
from experiments.E165_joint_l0.evaluation.repair.backend.guard import (
    repair_tiny_negative,
)
from experiments.E165_joint_l0.evaluation.repair.fast_geometry import (
    topology,
    step as frozen_step,
    contact_tree,
)
from experiments.E165_joint_l0.evaluation.repair.canonical import canonicalize
from experiments.E165_joint_l0.evaluation.repair.projection import solve as project
from experiments.E165_joint_l0.evaluation.repair.certify_polish import polish

from .speed_kernels import hard_valid, reduce_edges
from .speed_shape import step


def solve(
    draft,
    p,
    objective,
    *,
    depth_steps=None,
    mode=0,
    boundary=12.0,
    contact=6.0,
    robust=True,
    certified=True,
    periodic=False,
    reweighted=True
):
    start = time.perf_counter()
    bucket = 0 if p.n < 60 else 1 if p.n < 90 else 2 if p.n < 110 else 3
    cap = (
        BUDGET["buckets"][bucket]["main_steps"] if depth_steps is None else depth_steps
    )
    fallback = row_control(p)
    best = fallback.copy()
    value = objective(best)
    trace = []
    counters = dict(
        shape_calls=0, projection_calls=0, rescue_seeds=0, shape_native_iterations=0
    )
    widths, heights = _dimensions(p, draft[:, 2] / draft[:, 3], True)
    state = np.column_stack((draft[:, :2], widths, heights))

    def accept(x):
        nonlocal best, value
        if x is None:
            return None
        x = canonicalize(x, p, tolerance=1e-4)
        x, _ = repair_tiny_negative(x, p)
        if not hard_valid(x, p):
            return None
        score = objective(x)
        if score < value:
            best, value = x.copy(), score
        return x

    accept(draft)
    legal_steps = 0
    stale = 0
    infeasible = False

    def shape_once(iteration, label):
        nonlocal state, legal_steps, stale, infeasible
        full = topology(state, p.constraints, mode=mode, array=True)
        previous = value
        edges = reduce_edges(full, p.n)
        elastic = 100000.0 * (1 + iteration / 3)
        if objective.mibs:
            proposal, stats = frozen_step(
                p,
                state,
                edges,
                boundary=boundary,
                contact=contact,
                reweighted=reweighted,
                conservative=True,
                full_edges=full,
                elastic=elastic,
            )
        else:
            proposal, stats = step(
                p,
                state,
                edges,
                full,
                objective.nsoft,
                boundary=boundary,
                contact=contact,
                elastic=elastic,
                reweighted=reweighted,
            )
        counters["shape_calls"] += 1
        counters["shape_native_iterations"] += stats.get("iterations", 0)
        clean = accept(proposal)
        trace.append(dict(phase=label, legal=clean is not None, **stats))
        if proposal is None:
            infeasible = stats["status"] == "PrimalInfeasible"
            return False
        legal_steps += int(clean is not None)
        state = clean if clean is not None else proposal
        stale = stale + 1 if previous - value < 1e-5 else 0
        return not (
            clean is not None
            and stale >= 2
            and legal_steps >= 4
            and sum(objective.violations(best)) == 0
        )

    def projection_once(label, contacts=True):
        nonlocal state
        full = topology(state, p.constraints, mode=mode, array=True)
        tree = contact_tree(state, p, full) if contacts else []
        proposal, stats = project(
            p,
            state,
            reduce_edges(full, p.n),
            tree,
            boundary=boundary,
            contact=contact if contacts else 0.0,
            hard_tangent=periodic,
            quadratic=True,
        )
        counters["projection_calls"] += 1
        clean = accept(proposal)
        trace.append(dict(phase=label, legal=clean is not None, **stats))
        if clean is not None:
            state = clean

    for iteration in range(cap):
        proceed = shape_once(iteration, "shape")
        if periodic and iteration % 4 == 3 and not infeasible:
            projection_once("periodic_project")
        if not proceed:
            break
    if np.array_equal(best, fallback):
        projection_once("early_projection", contacts=False)
        if np.array_equal(best, fallback) and not infeasible:
            for iteration in range(BUDGET["extension_shape_steps"]):
                if not shape_once(cap + iteration, "extension_shape"):
                    break
    if not infeasible:
        if not np.array_equal(best, fallback):
            state = best.copy()
        projection_once("project")
    if robust and (np.array_equal(best, fallback) or objective(best) > 0.9):
        for seed_mode in BUDGET["rescue_seed_modes"]:
            counters["rescue_seeds"] += 1
            packed = seed_pack(draft, p, seed_mode)
            if accept(packed) is None:
                continue
            refined, stats = solve(
                packed,
                p,
                objective,
                depth_steps=BUDGET["rescue_shape_steps"],
                mode=1,
                boundary=40.0,
                contact=20.0,
                robust=False,
                certified=False,
                periodic=True,
                reweighted=False,
            )
            accept(refined)
            trace.append(
                dict(
                    phase="full_packed_refine",
                    mode=seed_mode,
                    work=stats["work"],
                    trace=stats["trace"],
                )
            )
            if objective(best) < 0.6:
                break
    if certified:
        best, stats = polish(
            best,
            p,
            max_accept=BUDGET["polish_max_accept"],
            max_candidates=BUDGET["polish_max_candidates"],
        )
        trace.append(dict(phase="certified_rigid_moves", **stats))
        value = objective(best)
    assert hard_valid(best, p)
    return best, dict(
        seconds=time.perf_counter() - start,
        proxy=value,
        trace=trace,
        work=counters,
        bucket=bucket,
        main_step_cap=cap,
        row_retained=bool(np.array_equal(best, fallback)),
        public_stopped=False,
        evaluator_revision=BUDGET["revision"],
    )
