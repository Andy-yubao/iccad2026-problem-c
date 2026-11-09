"""Explicit opt-in front end; the downstream planner remains separate."""
import time
import numpy as np
from .relaxation import relax


def preprocess(draft, inputs, *, config=None, return_history=False):
    n, _, _, _, _, constraints, specs = inputs
    p = np.asarray(draft, dtype=float)
    c, s = np.asarray(constraints), np.asarray(specs)
    if p.shape != (int(n), 4) or c.shape != (int(n), 5) or s.shape != p.shape:
        raise ValueError('draft and public specification dimensions differ')
    preplaced = c[:, 1] != 0
    locked_shape = (c[:, :2] != 0).any(axis=1)
    if not np.array_equal(p[locked_shape, 2:], s[locked_shape, 2:]) or not np.array_equal(p[preplaced], s[preplaced]):
        raise ValueError('restore public immutable geometry before preprocessing')
    return relax(p, preplaced, config=config, return_history=return_history)


def solve(draft, inputs, planner, *, config=None, planner_options=None):
    """Compose a supplied planner with E99, including both in wall time.

    The caller is responsible for final hard checks and timing any upstream NN
    and transfers. This duration starts from an already resident CPU draft.
    """
    start = time.perf_counter()
    revised, relaxation = preprocess(draft, inputs, config=config)
    final, downstream = planner(revised, inputs, **(planner_options or {}))
    return final, dict(relaxation=relaxation, planner=downstream,
                       cpu_draft_to_planner_seconds=time.perf_counter() - start)
