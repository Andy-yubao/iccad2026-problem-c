"""Explicit new-research scoring; original inputs and official code stay intact."""
import numpy as np

from .evaluation import evaluate_case as evaluate_original
from .official import official_evaluator

POLICY = "research.original_no_mib.v1"


def evaluate_case(case, positions, runtime_seconds):
    result = evaluate_original(case, positions, runtime_seconds)
    metrics = result["metrics"]
    original = dict(metrics)
    mib = np.asarray(case.solver_inputs[5])[:, 2]
    maximum = sum(max(0, int((mib == group).sum()) - 1)
                  for group in np.unique(mib) if group > 0)
    metrics["mib_violations"] = 0
    metrics["total_soft_violations"] -= original["mib_violations"]
    metrics["max_possible_violations"] -= maximum
    metrics["violations_relative"] = (metrics["total_soft_violations"] /
                                       max(metrics["max_possible_violations"], 1))
    metrics["cost"] = official_evaluator().compute_cost(
        metrics["hpwl_gap"], metrics["area_gap"], metrics["violations_relative"],
        1.0, metrics["is_feasible"])
    result["scoring_policy"] = POLICY
    result["original_official_metrics_audit_only"] = original
    return result
