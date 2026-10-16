"""Official neutral-runtime output scoring; actual solve time is reported separately.

This is the existing local quality comparison (runtime=median_runtime=1), not
a replacement for the contest median-runtime score and not a training loss.
"""
from dataclasses import asdict
import math
from scoring.official import official_evaluator

def score_official_output(case, positions, runtime_seconds, median_runtime):
    """Return raw official metrics using the supplied contest time baseline."""
    _, areas, b2b, p2b, pins, constraints, _ = case.solver_inputs
    return asdict(official_evaluator().evaluate_solution(
        {"positions": positions, "runtime": float(runtime_seconds)}, case.baseline_metrics,
        constraints, b2b, p2b, pins, areas, case.reference_positions,
        median_runtime=float(median_runtime)))

def evaluate_case(case, positions, runtime_seconds):
    """Use official scoring with runtime factor 1 and an independent hard check."""
    from scoring.checks import independent_hard_checks
    if hasattr(positions, "tolist"):
        positions = positions.tolist()
    positions = [[float(v) for v in row] for row in positions]
    if len(positions) != case.block_count or any(len(row) != 4 for row in positions):
        raise ValueError("Optimizer must return exactly block_count rows of XYWH")
    if any(not all(math.isfinite(v) for v in row) or min(row[2:]) <= 0 for row in positions):
        raise ValueError("Optimizer returned non-finite coordinates or nonpositive dimensions")
    n, areas, b2b, p2b, pins, constraints, _ = case.solver_inputs
    independent = independent_hard_checks(positions, areas, constraints, case.reference_positions)
    official = official_evaluator()
    # Scoring runtime=median_runtime=1 avoids the official 0.01s denominator floor.
    # Actual measured runtime is restored in the output field below.
    metrics = score_official_output(case, positions, 1.0, 1.0)
    metrics["runtime_seconds"] = float(runtime_seconds)
    if metrics["is_feasible"] != independent["is_feasible"]:
        raise AssertionError("Official and independent feasibility check disagree")
    if metrics["dimension_violations"] != independent["dimension_or_location_violation_count"]:
        raise AssertionError("Official and independent fixed/preplaced checks disagree")
    # Q&A requires nonnegative coordinates, but the archived evaluator and audit
    # omit this check.  Preserve its raw flag and explicitly enforce the input
    # contract in the research feasibility/cost rather than accepting a loophole.
    negative = [i for i, row in enumerate(positions) if row[0] < 0 or row[1] < 0]
    metrics["official_is_feasible_before_nonnegative_check"] = metrics["is_feasible"]
    metrics["nonnegative_coordinate_violations"] = len(negative)
    independent["negative_coordinate_blocks"] = negative
    if negative:
        independent["is_feasible"] = False
        metrics["is_feasible"] = False
        metrics["cost"] = official.M_PENALTY
    return {"metrics": metrics, "independent_hard_checks": independent, "positions": positions}


def summarize_variants(cases, strategies):
    """Failures count as infeasible with cost 10; undefined gaps remain explicit."""
    official = official_evaluator()
    summaries = {}
    for strategy in strategies:
        records = [case["variants"][strategy] for case in cases]
        metrics = [record["metrics"] for record in records]
        returned = [metric for metric in metrics if metric.get("hpwl_gap") is not None]
        def mean(key):
            return sum(metric[key] for metric in returned) / len(returned) if returned else None
        summaries[strategy] = {
            "case_count": len(cases), "feasible_count": sum(metric["is_feasible"] for metric in metrics),
            "feasible_rate": sum(metric["is_feasible"] for metric in metrics) / len(cases),
            "execution_error_count": sum("error" in record for record in records),
            "neutral_runtime_mean_cost_all_cases": sum(metric["cost"] for metric in metrics) / len(cases),
            "neutral_runtime_weighted_cost_all_cases": official.compute_total_score(
                [metric["cost"] for metric in metrics], [case["input_summary"]["block_count"] for case in cases]),
            "mean_hpwl_gap_returned_geometry": mean("hpwl_gap"),
            "mean_area_gap_returned_geometry": mean("area_gap"),
            "mean_boundary_violations_returned_geometry": mean("boundary_violations"),
            "mean_grouping_violations_returned_geometry": mean("grouping_violations"),
            "mean_mib_violations_returned_geometry": mean("mib_violations"),
            "returned_geometry_metric_count": len(returned),
            "total_solve_seconds": sum(record.get("solve_seconds", 0) for record in records),
            "total_worker_wall_seconds": sum(record.get("worker_wall_seconds", 0) for record in records),
            "max_peak_worker_memory_bytes": max([record.get("worker_peak_memory_bytes") or 0 for record in records]) or None,
            "quality_scope": "all returned geometries, including infeasible ones; errors have undefined gaps and cost 10",
        }
    return summaries
