"""Fixed-shape contact repair and conditional legalization for E165 drafts."""
from dataclasses import dataclass
import math
import time

import numpy as np

from experiments.E165_joint_l0.evaluation import solver as baseline
from experiments.E165_joint_l0.evaluation.repair.backend.frozen.feasible import Objective
from experiments.E165_joint_l0.evaluation.repair.backend.frozen.solver import hard_valid

from .speed_kernels import NativeObjective
from .speed_solver import solve


BOUNDARY_BITS = ((1, 0, False), (2, 0, True), (8, 1, False), (4, 1, True))
BOUNDARY_TOLERANCE = 1e-6
MIN_CONTACT_FRACTION = 0.001
MAX_CONTACT_TARGETS = 32
MAX_SOFT_PASSES = 3
MAX_PUSHED_BLOCKS = 4
MAX_PUSH_FRACTION = 0.06


@dataclass
class Incumbent:
    layout: np.ndarray
    violations: tuple
    area: float
    wire: float


def measure(layout, problem, objective):
    area, wire = baseline.raw_measures(layout, problem)
    return Incumbent(layout, objective.violations(layout), area, wire)


def metric_ratio(new, old):
    if old > 0:
        return new / old
    return 1.0 if new == 0 else math.inf


def certificate(challenger, incumbent, nsoft):
    """Upper-bound the neutral-runtime Q ratio for any reference denominators."""
    if any(new > old for new, old in zip(challenger.violations, incumbent.violations)):
        return math.inf
    growth = max(1.0, metric_ratio(challenger.area, incumbent.area),
                 metric_ratio(challenger.wire, incumbent.wire))
    soft_change = sum(challenger.violations) - sum(incumbent.violations)
    return growth * math.exp(2.0 * soft_change / nsoft)


def connected_components(layout, group):
    lower = layout[group, :2]
    upper = lower + layout[group, 2:]
    overlap = np.minimum(upper[:, None], upper[None, :]) - np.maximum(
        lower[:, None], lower[None, :])
    contact = (overlap >= 0).all(axis=2) & (overlap > 0).any(axis=2)
    unseen = set(range(len(group)))
    components = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack = [first]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            neighbors = unseen.intersection(np.flatnonzero(contact[index]))
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors))
        components.append(group[sorted(component)])
    return components


def free_tangent_shifts(lower, upper, axis, normal_shift, bounds, others):
    """Return nearby legal points in the complement of open collision intervals."""
    lo, hi = bounds
    if lo > hi:
        return []
    tangent = 1 - axis
    rest_lo, rest_hi = others
    cross = (
        np.minimum(upper[:, None, axis] + normal_shift, rest_hi[:, axis])
        - np.maximum(lower[:, None, axis] + normal_shift, rest_lo[:, axis])
    ) > 1e-8
    starts = (rest_lo[:, tangent] - upper[:, None, tangent])[cross]
    ends = (rest_hi[:, tangent] - lower[:, None, tangent])[cross]
    relevant = (starts < hi) & (ends > lo)
    starts, ends = starts[relevant], ends[relevant]
    cursor = lo
    available = []
    for index in np.argsort(starts, kind='stable'):
        start, end = starts[index], ends[index]
        if start >= cursor:
            available.append(float(np.clip(0.0, cursor, start)))
        cursor = max(cursor, end)
        if cursor > hi:
            break
    if cursor <= hi:
        available.append(float(np.clip(0.0, cursor, hi)))
    return sorted(available, key=abs)[:3]


def moved_layout(layout, ids, shift):
    result = layout.copy()
    result[ids, :2] += shift
    return result


def push_blockers(layout, candidate, ids, axis, direction, problem):
    """Open a slot by moving a bounded number of movable blockers."""
    if (candidate[ids, :2] < 0).any():
        return None
    fixed = problem.preplaced.copy()
    fixed[ids] = True
    movable = np.flatnonzero(~np.isin(np.arange(problem.n), ids))
    pushed = set()
    current = list(ids)
    for _ in range(MAX_PUSHED_BLOCKS + 1):
        changed = set()
        for i in current:
            a = candidate[i]
            low = candidate[movable, :2]
            high = low + candidate[movable, 2:]
            overlap = np.minimum(a[:2] + a[2:], high) - np.maximum(a[:2], low)
            for j in movable[(overlap > 1e-8).all(axis=1) & (movable != i)]:
                # A pushed block must keep moving away from the initiating slot.
                if i in pushed and direction * (candidate[j, axis] - a[axis]) < 0:
                    continue
                if fixed[j]:
                    return None
                if direction == 1:
                    target = a[axis] + a[axis + 2]
                else:
                    target = a[axis] - candidate[j, axis + 2]
                if target < 0 or abs(target - layout[j, axis]) > problem.scale * MAX_PUSH_FRACTION:
                    return None
                candidate[j, axis] = target
                changed.add(int(j))
                pushed.add(int(j))
                if len(pushed) > MAX_PUSHED_BLOCKS:
                    return None
        if not changed:
            return candidate
        current = sorted(changed)
    return None


def translations(layout, ids, axis, normal_shift, tangent_window, problem, push_direction):
    lower = layout[:, :2]
    upper = lower + layout[:, 2:]
    other = np.ones(problem.n, dtype=bool)
    other[ids] = False
    tangent = 1 - axis
    if (lower[ids, axis] + normal_shift < 0).any():
        return
    lo, hi = tangent_window
    lo = max(lo, -float(lower[ids, tangent].min()))
    if lo > hi:
        return
    shifts = free_tangent_shifts(lower[ids], upper[ids], axis, normal_shift, (lo, hi),
                                (lower[other], upper[other]))
    shift = np.zeros(2)
    shift[axis] = normal_shift
    for tangent_shift in shifts:
        shift[tangent] = tangent_shift
        yield moved_layout(layout, ids, shift)
    if not shifts:
        shift[tangent] = np.clip(0.0, lo, hi)
        candidate = push_blockers(layout, moved_layout(layout, ids, shift), ids,
                                  axis, push_direction, problem)
        if candidate is not None:
            yield candidate


def boundary_proposals(layout, groups, objective, problem):
    lower = layout[:, :2]
    upper = lower + layout[:, 2:]
    lo, hi = lower.min(axis=0), upper.max(axis=0)
    by_block = {int(i): component for component in groups for i in component}
    for i in np.flatnonzero(objective.codes):
        requested = [(axis, upper_side) for bit, axis, upper_side in BOUNDARY_BITS
                     if objective.codes[i] & bit]
        shift = {a: hi[a] - upper[i, a] if side else lo[a] - lower[i, a]
                 for a, side in requested}
        missed = [(a, side) for a, side in requested if abs(shift[a]) >= BOUNDARY_TOLERANCE]
        if not missed:
            continue
        if problem.preplaced[i]:
            for proposal in preplaced_boundary(layout, i, missed, problem):
                yield proposal
            continue
        component = by_block.get(int(i), np.array([i]))
        movers = [component] if len(component) == 1 else [component, np.array([i])]
        for ids in movers:
            if problem.preplaced[ids].any():
                continue
            for axis, upper_side in missed:
                t = 1 - axis
                window = (lo[t] - lower[ids, t].min(), hi[t] - upper[ids, t].max())
                if t in shift:
                    window = (shift[t], shift[t])
                yield from translations(layout, ids, axis, shift[axis], window, problem,
                                        -1 if upper_side else 1)


def preplaced_boundary(layout, anchor, missed, problem):
    """Move the few offending blocks when the boundary block is immovable."""
    candidate = layout
    for axis, upper_side in missed:
        edge = candidate[:, axis].copy()
        if upper_side:
            edge += candidate[:, axis + 2]
        distance = edge[anchor] - edge
        protrudes = distance < -BOUNDARY_TOLERANCE if upper_side else distance > BOUNDARY_TOLERANCE
        offenders = np.flatnonzero(protrudes)
        if len(offenders) > MAX_PUSHED_BLOCKS or problem.preplaced[offenders].any():
            return
        for i in sorted(offenders, key=lambda j: -abs(distance[j])):
            tangent = 1 - axis
            lo = candidate[:, tangent].min() - candidate[i, tangent]
            hi = (candidate[:, tangent] + candidate[:, tangent + 2]).max()
            hi -= candidate[i, tangent] + candidate[i, tangent + 2]
            proposals = translations(candidate, np.array([i]), axis, distance[i], (lo, hi),
                                     problem, -1 if upper_side else 1)
            candidate = next(proposals, None)
            if candidate is None:
                return
    yield candidate


def group_proposals(layout, group_components, problem):
    lower = layout[:, :2]
    upper = lower + layout[:, 2:]
    lo, hi = lower.min(axis=0), upper.max(axis=0)
    for components in group_components:
        if len(components) == 1:
            continue
        group = np.concatenate(components)
        for ids in components:
            if problem.preplaced[ids].any():
                continue
            rest = group[~np.isin(group, ids)]
            candidates = []
            for axis in (0, 1):
                t = 1 - axis
                i, j = np.meshgrid(ids, rest, indexing='ij')
                tangent_overlap = MIN_CONTACT_FRACTION * np.minimum(layout[i, t + 2], layout[j, t + 2])
                left = np.maximum(lower[j, t] - upper[i, t] + tangent_overlap,
                                  lo[t] - lower[ids, t].min())
                right = np.minimum(upper[j, t] - lower[i, t] - tangent_overlap,
                                   hi[t] - upper[ids, t].max())
                possible = left <= right
                guide = np.abs(np.clip(0.0, left, right))
                for shifts, direction in ((lower[j, axis] - upper[i, axis], -1),
                                          (upper[j, axis] - lower[i, axis], 1)):
                    for normal, lower_t, upper_t, lateral in zip(
                            shifts[possible], left[possible], right[possible], guide[possible]):
                        distance = abs(normal) + lateral
                        candidates.append((distance, axis, normal, lower_t, upper_t, direction))
            for _, axis, normal, lo_t, hi_t, direction in sorted(candidates)[:MAX_CONTACT_TARGETS]:
                yield from translations(layout, ids, axis, normal, (lo_t, hi_t), problem, direction)


def choose(layout, incumbent, problem, objective):
    if layout is None or not hard_valid(layout, problem):
        return incumbent, (1.0, 0.0)
    candidate = measure(layout, problem, objective)
    bound = certificate(candidate, incumbent, objective.nsoft)
    improvement = (1.0 - metric_ratio(candidate.area, incumbent.area)
                   + 1.0 - metric_ratio(candidate.wire, incumbent.wire))
    # For equal soft counts a Pareto improvement in geometry is sufficient.
    if sum(candidate.violations) == sum(incumbent.violations):
        ratios = (metric_ratio(candidate.area, incumbent.area),
                  metric_ratio(candidate.wire, incumbent.wire))
        if bound <= 1.0 + 1e-12 and min(ratios) < 1.0 - 1e-9:
            return candidate, (1.0, -improvement)
    elif bound < 1.0 - 1e-9:
        return candidate, (bound, -improvement)
    return incumbent, (1.0, 0.0)


def refine(draft, inputs, original, *, legal_main, prepared=None,
           search=baseline.optimize):
    """Refine an existing legal E165 output using public inputs only."""
    start = time.perf_counter()
    if prepared is None:
        problem = baseline.prepare_problem(*inputs)
        objective = Objective(problem)
    else:
        problem, objective = prepared
    current = measure(original, problem, objective)
    info = dict(soft_moves=0, legal_seed_attempted=False, legal_seed_accepted=False,
                anchor_seed_attempted=False, anchor_seed_accepted=False)
    poor_tail = not legal_main and current.area > 1.10 * problem.total_area
    if sum(current.violations) or poor_tail:
        # A legal incumbent gives a sound topology to a shape-changing retry.
        info['legal_seed_attempted'] = True
        proposal, _ = search(current.layout, inputs, depth_steps=4,
                             mode=0, boundary=35., contact=40.,
                             robust=False, certified=True, periodic=True)
        updated, _ = choose(proposal, current, problem, objective)
        info['legal_seed_accepted'] = updated is not current
        current = updated
        if sum(current.violations) and not info['legal_seed_accepted']:
            # Public anchors can change the topology of the original draft.
            anchored = draft.copy()
            anchored[problem.preplaced] = problem.specs[problem.preplaced]
            info['anchor_seed_attempted'] = True
            proposal, _ = search(anchored, inputs, depth_steps=4,
                                 mode=1, boundary=40., contact=24.,
                                 robust=False, certified=True, periodic=True)
            updated, _ = choose(proposal, current, problem, objective)
            info['anchor_seed_accepted'] = updated is not current
            current = updated
    for _ in range(MAX_SOFT_PASSES):
        if not sum(current.violations):
            break
        groups = [connected_components(current.layout, group) for group in objective.groups]
        best, best_key = current, (1.0, 0.0)
        proposals = (
            boundary_proposals(current.layout, [c for group in groups for c in group], objective, problem)
            if current.violations[0] else ()
        )
        group_moves = group_proposals(current.layout, groups, problem) if current.violations[1] else ()
        for proposals_for_move in (proposals, group_moves):
            for proposal in proposals_for_move:
                candidate, key = choose(proposal, current, problem, objective)
                if key < best_key:
                    best, best_key = candidate, key
        if best is current:
            break
        current = best
        info['soft_moves'] += 1
    info['seconds'] = time.perf_counter() - start
    return current.layout, info


def optimize_reference(draft, inputs):
    """Run the frozen baseline and keep it as the incumbent throughout repair."""
    original, baseline_info = baseline.optimize(draft, inputs)
    legal_main = any(step.get('legal') and step['phase'] == 'shape'
                     for step in baseline_info['trace'])
    layout, extra = refine(draft, inputs, original, legal_main=legal_main)
    return layout, dict(seconds=baseline_info['seconds'] + extra['seconds'],
                        baseline=baseline_info, refinement=extra)


def optimize(draft, inputs):
    """Run the full postprocessor with the accelerated geometry kernels."""
    start = time.perf_counter()
    problem = baseline.prepare_problem(*inputs)
    objective = NativeObjective(problem)
    original, baseline_info = solve(draft, problem, objective)
    legal_main = any(
        step.get("legal") and step["phase"] == "shape"
        for step in baseline_info["trace"]
    )

    def search(seed, _inputs, **kwargs):
        return solve(seed, problem, objective, **kwargs)

    layout, extra = refine(
        draft,
        inputs,
        original,
        legal_main=legal_main,
        prepared=(problem, objective),
        search=search,
    )
    return layout, dict(
        seconds=time.perf_counter() - start, baseline=baseline_info, refinement=extra
    )
