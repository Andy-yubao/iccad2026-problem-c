"""Summarize the paired comparison with the saved E151 evaluation."""
import json

import numpy as np

from ..io import ROOT, save


def summarize(output, records, plan, frozen, evaluator):
    monitor = set(plan["monitor500_indices"])
    panels = {
        "monitor500": [row for row in records if row["index"] in monitor],
        "extension1500": [row for row in records if row["index"] not in monitor],
        "full2000": records,
    }
    subsets = {}
    for name, rows in panels.items():
        costs = [row["metrics"]["cost"] for row in rows]
        weighted = evaluator.compute_total_score(costs, [row["n"] for row in rows])
        mean = float(np.mean(costs))
        subsets[name] = dict(
            count=len(rows), weighted_Q=weighted, mean_Q=mean, mixed_Q=(weighted + mean) / 2,
            hard_legal=sum(row["independent_hard_checks"]["is_feasible"] for row in rows),
            grouping_satisfied=sum(row["metrics"]["grouping_violations"] == 0 for row in rows),
            boundary_satisfied=sum(row["metrics"]["boundary_violations"] == 0 for row in rows),
            all_soft_satisfied=sum(row["metrics"]["total_soft_violations"] == 0 for row in rows),
        )
    baseline_output = ROOT / plan["baseline_output"]
    baseline = json.loads((baseline_output / "subsets.json").read_text(encoding="utf-8"))
    differences = []
    geometry = {key: [] for key in ("L8_overlap", "L10_overlap", "L12_overlap")}
    for row in records:
        case_path = baseline_output / "cases" / f"{row['index']:04d}.json"
        original = json.loads(case_path.read_text(encoding="utf-8"))
        if (original["member"] != row["member"]
                or original["checkpoint_sha256"] != frozen["checkpoint_sha256"]
                or original["L8_sha256"] != row["L8_sha256"]):
            raise ValueError(f"E151 checkpoint, member or L8 mismatch at case {row['index']}")
        differences.append(row["metrics"]["cost"] - original["metrics"]["cost"])
        for key in geometry:
            geometry[key].append(row["s2_geometry"][key] - original["s2_geometry"][key])
    deltas = np.asarray(differences)
    save(output / "subsets.json", subsets)
    save(output / "comparison.json", dict(
        baseline=plan["baseline_output"],
        paired_count=len(deltas),
        identical_L8_count=len(deltas),
        improved=int((deltas < -1e-9).sum()),
        tied=int((abs(deltas) <= 1e-9).sum()),
        worsened=int((deltas > 1e-9).sum()),
        mixed_Q_change={name: row["mixed_Q"] - baseline[name]["mixed_Q"] for name, row in subsets.items()},
        mean_overlap_change={key: float(np.mean(values)) for key, values in geometry.items()},
    ))
    full = subsets["full2000"]
    return {key: full[key] for key in ("grouping_satisfied", "boundary_satisfied", "all_soft_satisfied")}
