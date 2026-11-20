"""Observational wrappers around the unmodified local E7 control."""
from collections import defaultdict
import time
import numpy as np
from .frozen import global_refine, shape_lp, solver
from .frozen.feasible import Objective


def invalid_reasons(pos, p):
    a = np.asarray(pos)
    if a.shape != (p.n, 4) or not np.isfinite(a).all():
        return {'invalid_array': True}
    locked = p.fixed | p.preplaced
    free = ~locked
    i, j = np.triu_indices(p.n, 1)
    overlap = np.minimum(a[i, :2] + a[i, 2:], a[j, :2] + a[j, 2:]) - np.maximum(a[i, :2], a[j, :2])
    return {
        'negative_coordinate_count': int((a[:, :2] < 0).sum()),
        'min_coordinate': float(a[:, :2].min()),
        'nonpositive_dimension_count': int((a[:, 2:] <= 0).sum()),
        'locked_dimensions': int((np.abs(a[locked, 2:] - p.specs[locked, 2:]) > 1e-8).sum()),
        'preplaced': int((np.abs(a[p.preplaced] - p.specs[p.preplaced]) > 1e-8).sum()),
        'area': int((np.abs(a[free, 2:].prod(1) - p.areas[free]) / p.areas[free] > .01).sum()),
        'overlap_pairs': int((overlap > 1e-6).all(1).sum()),
    }


class Observe:
    def __init__(self):
        self.begin = time.perf_counter()
        self.time = defaultdict(float)
        self.count = defaultdict(int)
        self.events = []
        self.disagreements = []
        self.saved = []
        self.best = float('inf')
        self.phase = 'initial'
        self.pre_guard = None

    def patch(self, obj, key, value):
        self.saved.append((obj, key, getattr(obj, key)))
        setattr(obj, key, value)

    def __enter__(self):
        parent = self
        for obj, key, phase in ((global_refine, 'solve_lp', 'fixed_lp'),
                                (shape_lp, 'step', 'shape'),
                                (global_refine, 'finalize', 'initial_pack'),
                                (solver, 'finalize', 'guard_pack'),
                                (global_refine, 'numerical_cleanup', 'cleanup')):
            original = getattr(obj, key)
            def wrapped(*args, _original=original, _phase=phase, **kwargs):
                before = time.perf_counter()
                previous = parent.phase
                parent.phase = _phase
                try:
                    return _original(*args, **kwargs)
                finally:
                    parent.phase = previous
                    parent.time[_phase] += time.perf_counter() - before
                    parent.count[_phase] += 1
            self.patch(obj, key, wrapped)
        original_linprog = global_refine.linprog
        def lp(*args, **kwargs):
            before = time.perf_counter()
            try:
                return original_linprog(*args, **kwargs)
            finally:
                parent.time['fixed_solver_only'] += time.perf_counter() - before
        self.patch(global_refine, 'linprog', lp)
        original_highs = shape_lp.highspy.Highs
        class TimedHighs(original_highs):
            def run(self):
                before = time.perf_counter()
                try:
                    return super().run()
                finally:
                    parent.time['shape_solver_only'] += time.perf_counter() - before
                    parent.count['shape_solver_calls'] += 1
        self.patch(shape_lp.highspy, 'Highs', TimedHighs)
        original_objective = Objective.__call__
        def objective(obj, pos):
            value = original_objective(obj, pos)
            # Only observe; never feed official Q/reference information to solver.
            if solver.hard_valid(pos, obj.p) and value < parent.best - 1e-10:
                parent.best = value
                parent.events.append({'elapsed': time.perf_counter() - parent.begin,
                                      'proxy': float(value), 'positions': np.asarray(pos).tolist()})
            return value
        self.patch(Objective, '__call__', objective)
        old_legal = global_refine.is_legal
        def legal(pos, problem):
            good = old_legal(pos, problem)
            if good and not solver.hard_valid(pos, problem):
                parent.disagreements.append(invalid_reasons(pos, problem))
            return good
        self.patch(global_refine, 'is_legal', legal)
        original_refine=solver.refine
        def refinement(*args,**kwargs):
            candidate,stats=original_refine(*args,**kwargs)
            problem=global_refine.prepare_problem(*args[1])
            good=solver.hard_valid(candidate,problem)
            parent.pre_guard={'strict_valid':good,'reasons':invalid_reasons(candidate,problem)}
            if not good and np.isfinite(np.asarray(candidate)).all():
                parent.pre_guard['rejected_positions']=np.asarray(candidate).tolist()
            return candidate,stats
        self.patch(solver,'refine',refinement)
        return self

    def __exit__(self, *unused):
        for obj, key, original in reversed(self.saved):
            setattr(obj, key, original)

    def result(self):
        return {'seconds_by_component': dict(self.time), 'calls': dict(self.count),
                'loose_strict_disagreements': self.disagreements, 'pre_guard_validation':self.pre_guard,
                'improvements': self.events}
