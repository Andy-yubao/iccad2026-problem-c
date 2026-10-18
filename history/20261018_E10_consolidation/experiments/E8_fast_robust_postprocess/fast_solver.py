"""Route A pilot: fewer precedence edges and adaptive area cuts, strict outputs."""
import time
import numpy as np
from .frozen.packing import prepare_problem, _dimensions
from .frozen.feasible import Objective, row_control
from .frozen.postprocess import finalize
from .frozen.global_refine import topology, contact_tree, solve_lp
from .frozen.canonical_geometry import canonicalize
from .frozen.solver import hard_valid
from .reduction import reduce_edges
from .sparse_shape import step


def optimize(raw, inputs, seconds=1.5, reduced=True):
    begin = time.perf_counter()
    deadline = begin + seconds
    problem = prepare_problem(*inputs)
    objective = Objective(problem)
    trace, events = [], []
    best = row_control(problem)
    if not hard_valid(best, problem):
        raise RuntimeError('Public specifications have no valid row fallback')
    value = objective(best)
    events.append({'elapsed': time.perf_counter()-begin, 'proxy': value, 'positions': best.tolist()})
    rejects = 0
    def accept(candidate, label):
        nonlocal best, value, rejects
        if candidate is None:
            return False
        candidate = canonicalize(candidate, problem)
        # Repair only tiny numerical negative coordinates, then recheck all hard rules.
        coordinates = candidate[:, :2]
        coordinates[(coordinates < 0) & (coordinates >= -1e-9)] = 0.
        if not hard_valid(candidate, problem):
            rejects += 1
            return False
        current = objective(candidate)
        if current < value - 1e-10:
            best, value = candidate.copy(), current
            events.append({'elapsed': time.perf_counter()-begin, 'proxy': float(value),
                           'phase': label, 'positions': best.tolist()})
        return True
    if hard_valid(raw, problem):
        accept(np.asarray(raw).copy(), 'raw')
    if deadline-time.perf_counter() > .08:
        before=time.perf_counter()
        seed, _ = finalize(raw, inputs, min(.12, max(.01, deadline-time.perf_counter()-.04)))
        accept(seed, 'compact_pack')
        trace.append({'phase':'compact_pack','seconds':time.perf_counter()-before})
    dims = _dimensions(problem, np.asarray(raw)[:,2]/np.asarray(raw)[:,3], True)
    state = np.column_stack((np.asarray(raw)[:,:2], *dims))
    for mode in (0,1):
        remaining=deadline-time.perf_counter()-.035
        if remaining < .04:
            break
        before=time.perf_counter()
        full=topology(state,problem.constraints,mode)
        edges=reduce_edges(full,problem.n) if reduced else full
        tree=contact_tree(state,problem,full)
        candidate, stats=solve_lp(problem,state,edges,contacts=tree,contact=8.,
                                 limit=min(.18,remaining))
        legal=accept(candidate,'fixed_lp')
        trace.append({'phase':'fixed_lp','mode':mode,'seconds':time.perf_counter()-before,
                      'full_edges':len(full),'edges':len(edges),'legal':legal,**stats})
    # Start shape correction from the draft, rather than locking into a bad fallback.
    for iteration in range(4):
        remaining=deadline-time.perf_counter()-.035
        if remaining < .05:
            break
        before=time.perf_counter()
        full=topology(state,problem.constraints)
        edges=reduce_edges(full,problem.n) if reduced else full
        candidate,stats=step(problem,state,edges,seconds=remaining,reweighted=True,
                             elastic=100000*(1+iteration/3),iteration=iteration,
                             tangents=5 if reduced else 25)
        legal=accept(candidate,'shape')
        trace.append({'phase':'shape','iteration':iteration,'seconds':time.perf_counter()-before,
                      'full_edges':len(full),'edges':len(edges),'legal':legal,**stats})
        if candidate is None:
            break
        state=candidate
    if not hard_valid(best,problem):
        raise AssertionError('Strict incumbent contract violated')
    return best, {'seconds':time.perf_counter()-begin,'proxy':float(value),'trace':trace,
                  'strict_rejects':rejects,'guard_fallback':False,'improvements':events}
