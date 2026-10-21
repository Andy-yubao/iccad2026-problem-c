"""Selected E8 postprocessor. Inputs are the public problem tuple and E7 draft.

Call optimize(draft_xywh, masked_solver_inputs) -> (hard_legal_xywh, stats).
Ordinary cases use 1.3 seconds; only missing normal candidates trigger rescue.
Five seconds is the total target budget, including rescue and final checking.
"""
from .robust import optimize as _optimize


def optimize(draft,solver_inputs,seconds=5.):
    return _optimize(draft,solver_inputs,seconds=seconds,fast_seconds=1.3)
