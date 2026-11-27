"""E44 input-only stage-A solver; no learned predictor is claimed yet."""
from dataclasses import dataclass
import time
import numpy as np
from .problem import Problem, dimensions, legal_incumbent, hard_check, public_objective
from .structure import propose, Structure
from .geometry import solve_fixed


@dataclass(frozen=True)
class Budget:
    seconds: float = .18
    max_lp_calls: int = 4
    shape_updates: int = 2
    threads: int = 4

    def validate(self):
        if not np.isfinite(self.seconds) or self.seconds <= 0 or not 1 <= self.max_lp_calls <= 8:
            raise ValueError('Invalid stage-A time/call budget')
        if not 0 <= self.shape_updates <= 3 or not 1 <= self.threads <= 4:
            raise ValueError('Invalid stage-A shape/thread budget')
        return self


def shape_step(p, pos):
    """One bounded analytic shape step; same topology and acceptance objective.

    This fixed rule is a control, not a learned shape optimizer. Derivatives hold
    XY fixed; the LP recomputes geometry and true public quality after the step.
    """
    wh, center = pos[:, 2:], pos[:, :2] + pos[:, 2:] / 2
    grad_xy = np.zeros((p.n, 2))
    norm = max(float(p.b2b[:, 2].sum() + p.p2b[:, 2].sum()), 1e-12)
    for i, j, w in p.b2b:
        i, j = int(i), int(j)
        derivative = .5 * w / (norm * p.scale) * np.sign(center[i] - center[j])
        grad_xy[i] += derivative; grad_xy[j] -= derivative
    for pin, i, w in p.p2b:
        i, pin = int(i), int(pin)
        grad_xy[i] += .5 * w / (norm * p.scale) * np.sign(center[i] - p.pins[pin])
    grad = grad_xy[:, 0] * wh[:, 0] / 4 - grad_xy[:, 1] * wh[:, 1] / 4
    hi, lo = (pos[:, :2] + wh).max(0), pos[:, :2].min(0)
    extent = hi - lo
    for a in (0, 1):
        ids = np.flatnonzero(np.abs(pos[:, a] + wh[:, a] - hi[a]) < 1e-9)
        grad[ids] += .25 * extent[1 - a] * wh[ids, a] / p.areas.sum() / len(ids) * (1 if a == 0 else -1)
    r = np.log(wh[:, 0] / wh[:, 1])
    for group in set(p.constraints[:, 2]) - {0}:
        ids = np.flatnonzero(p.constraints[:, 2] == group)
        grad[ids] = grad[ids].sum()
        if p.locked[ids].any():
            grad[ids] = 0.
    grad[p.locked] = 0.
    r -= .08 * np.sign(grad)
    return dimensions(p, r)[0]


def optimize(p, incumbent, structure, *, budget=Budget(), contact_mode='exact'):
    """Optimize a supplied structure, retracting lowest-confidence soft decisions.

    Complete separation, anchor and area requirements are never retracted. If
    even that base topology is infeasible, retain the checked original incumbent.
    """
    budget.validate(); structure.validate()
    started = time.perf_counter()
    if not hard_check(p, incumbent)['is_feasible']:
        raise ValueError('A hard-legal safety incumbent is required')
    best = incumbent.copy()
    score = public_objective(p, best)
    initial = dict(score)
    active = sorted(structure.decisions, key=lambda d: (d.confidence, d.kind, d.i, d.j, d.bit))
    attempts, retracted, accepted, updates = [], [], 0, 0
    wh = best[:, 2:].copy()
    fixed_feasible = False
    for call in range(budget.max_lp_calls):
        remaining = budget.seconds - (time.perf_counter() - started)
        if remaining <= .002:
            break
        candidate, stats = solve_fixed(p, best, wh, structure, active,
                                      seconds=remaining, threads=budget.threads,
                                      contact_mode=contact_mode)
        stats.update(call=call, shape_update=updates)
        attempts.append(stats)
        if candidate is None:
            if updates:
                # A rejected shape step cannot erase the prior feasible geometry.
                break
            if not active:
                break
            # Bounded search, never dozens of full solves. Ensure the final
            # allowed attempt can check the base structure without soft decisions.
            count = len(active) if call >= budget.max_lp_calls - 2 else max(1, (len(active) + 1) // 2)
            retracted.extend(active[:count]); active = active[count:]
            continue
        fixed_feasible = True
        actual = public_objective(p, candidate)
        stats['candidate_public_objective'] = actual
        stats['accepted'] = actual['value'] < score['value'] - 1e-10
        if stats['accepted']:
            best, score = candidate, actual
            accepted += 1
        if updates >= budget.shape_updates or not stats['accepted']:
            break
        updates += 1
        wh = shape_step(p, best)
    seconds = time.perf_counter() - started
    return best, {'origin': structure.origin, 'contact_mode': contact_mode,
                  'original_decision_count': len(structure.decisions),
                  'fixed_structure_feasible': fixed_feasible,
                  'initial_public_objective': initial, 'final_public_objective': score,
                  'lp_calls': len(attempts), 'attempts': attempts, 'accepted_updates': accepted,
                  'retracted': [d.__dict__ for d in retracted],
                  'remaining_decisions': [d.__dict__ for d in active],
                  'fallback_to_incumbent': not accepted, 'shape_updates_attempted': updates,
                  'optimizer_seconds': seconds, 'budget_seconds': budget.seconds,
                  'budget_overrun_seconds': max(0., seconds - budget.seconds),
                  'hard_valid': hard_check(p, best)['is_feasible'], 'uses_reference_objective': False}


class StructuralSolver:
    """Public contest-style input boundary, with explicit diagnostic metadata."""
    def __init__(self, *, budget=Budget(), contact_mode='exact', sequence=False):
        self.budget = budget.validate()
        if contact_mode not in ('exact', 'soft'):
            raise ValueError('Unknown contact mode')
        self.contact_mode, self.sequence = contact_mode, sequence
        self.last_info = None

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        started = time.perf_counter()
        if target_positions is None:
            target_positions = np.full((block_count, 4), -1.)
        p = Problem.from_inputs((block_count, area_targets, b2b_connectivity, p2b_connectivity,
                                 pins_pos, constraints, target_positions))
        wh, notes = dimensions(p)
        incumbent = legal_incumbent(p, wh)
        structure = propose(p, incumbent, use_sequence_pair=self.sequence)
        # All input processing and the safety path consume the same full budget.
        available = self.budget.seconds - (time.perf_counter() - started)
        if available > .002:
            local = Budget(available, self.budget.max_lp_calls, self.budget.shape_updates, self.budget.threads)
            result, info = optimize(p, incumbent, structure, budget=local, contact_mode=self.contact_mode)
        else:
            result, info = incumbent, {'fallback_to_incumbent': True, 'lp_calls': 0,
                                       'status': 'input_and_incumbent_budget_exhausted'}
        # The final hard certification is included in measured complete latency.
        if not hard_check(p, result)['is_feasible']:
            raise RuntimeError('E44 final certification failed')
        elapsed = time.perf_counter() - started
        self.last_info = {**info, 'mib_conflicts': notes, 'full_seconds': elapsed,
                          'requested_full_budget_seconds': self.budget.seconds,
                          'full_budget_overrun_seconds': max(0., elapsed - self.budget.seconds),
                          'process_initialization_included': False,
                          'lazy_solver_initialization_included': True, 'trained_predictor': False}
        return result.tolist()
