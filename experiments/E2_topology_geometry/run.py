"""Bounded E2 dev5 controls and targeted synthetic contradictions; no train/test scan."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def source_record(output, name="source.zip"):
    files = sorted(Path(__file__).parent.glob("*.py"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = output / name
    if not path.exists():
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in files:
                archive.write(source, source.relative_to(ROOT))
    else:
        with zipfile.ZipFile(path) as archive:
            for source in files:
                old = archive.read(source.relative_to(ROOT).as_posix())
                if hashlib.sha256(old).hexdigest() != hashes[source.name]:
                    raise ValueError("Existing source snapshot differs; preserve it and start a new run")
    return {"snapshot": path.name, "hashes": hashes}


def counterexamples():
    import numpy as np
    import torch
    from experiments.topology_geometry.geometry import DifferenceGraph, fixed_shapes, hard_valid, Optimizer
    from experiments.topology_geometry.input_geometry import prepare_problem
    from experiments.topology_geometry.model import RelationModel, features, ambiguous_targets, ambiguity_loss
    checks = {}
    graph = DifferenceGraph(np.ones(3), np.full(3, -1.), np.zeros(3, bool))
    assert graph.insert(0, 1) and graph.insert(1, 2) and not graph.insert(2, 0)
    checks["three_block_cycle_rejected"] = True
    # Locked blocks at x=0 and x=3 of width 1 leave a gap 2; a width-3
    # movable block cannot be squeezed into it by changing either anchor.
    graph = DifferenceGraph(np.array([1., 3., 1.]), np.array([0., -1., 3.]), np.array([True, False, True]))
    assert graph.insert(0, 1) and not graph.insert(1, 2)
    checks["impossible_gap_between_anchors_rejected"] = True
    areas = np.array([8., 2., 4.])
    constraints = np.array([[1, 0, 0, 0, 3], [0, 1, 0, 0, 1], [0, 0, 0, 0, 4]])
    specs = np.array([[-1., -1., 4., 2.], [2., 0., 1., 2.], [-1., -1., -1., -1.]])
    args = (3, areas, np.array([[0., 1., 1.], [1., 2., 1.]]), np.empty((0, 3)), np.empty((0, 2)), constraints, specs)
    p = prepare_problem(*args)
    dims = fixed_shapes(p)
    torch.manual_seed(0)
    for strategy in ("heuristic", "random"):
        solver = Optimizer(strategy=strategy, seed=0, seconds=2.)
        output = solver.solve(*args)
        assert hard_valid(p, output)
        assert list(output[0, 2:]) == [4., 2.]
        assert list(output[1]) == [2., 0., 1., 2.]
    checks["fixed_nonsquare_and_preplaced_specs_unchanged"] = True
    checks["contradictory_soft_left_right_request_does_not_change_shape"] = True
    # Deliberately poison every forbidden field. Sanitization must make the
    # solver inputs identical for ordinary positions/shapes and fixed XY.
    poisoned = specs.copy()
    poisoned[0, :2] = [12345., 67890.]
    poisoned[2] = [9999., 8888., 700., 600.]
    clean = Optimizer().solve(*args)
    poison = Optimizer().solve(*args[:-1], poisoned)
    assert np.array_equal(clean, poison)
    checks["forbidden_reference_fields_cannot_affect_solution"] = True
    reference = np.array([[0., 0., 4., 2.], [5., 0., 1., 2.], [7., 3., 2., 2.]])
    valid = ambiguous_targets(reference)
    assert valid[-1].sum() == 2
    model = RelationModel()
    data = features(p, dims)
    loss = ambiguity_loss(model(*data), valid)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    checks["ambiguous_multi_direction_target_has_finite_gradient"] = True
    checks["synthetic_loss"] = float(loss.detach())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15.)
    parser.add_argument("--total-seconds", type=float, default=90.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--counterexamples-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 15 or not 0 < args.total_seconds <= 180 or not 1 <= args.limit <= 5:
        parser.error("Bounded run requires seconds <=15, total <=180, 1..5 frozen dev cases")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import torch
    from datasets.cases import load_frozen_cases
    from datasets.paths import resolved_data_paths
    from scoring.evaluation import evaluate_case, summarize_variants
    from experiments.topology_geometry.geometry import Optimizer
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = source_record(args.output)
    config = {"command": sys.argv, "date_utc": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "panel": "frozen_dev_smoke_panel", "limit": args.limit,
              "seconds_per_case": args.seconds, "total_solve_seconds_budget": args.total_seconds,
              "threads": 1, "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "data_paths": resolved_data_paths(), "source": source,
              "split_sha256": hashlib.sha256((ROOT / "protocol/split_v1.json").read_bytes()).hexdigest(),
              "source_origin": "input_geometry.py prepare_problem copied from baseline/packing.py, independently maintained",
              "checkpoint": None, "learned_status": "not run: no trained checkpoint"}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    config["git_head"] = git.stdout.strip() if git.returncode == 0 else None
    if args.checkpoint:
        config.update(checkpoint=str(args.checkpoint.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      learned_status="trained checkpoint supplied")
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write(args.output / "config.json", config)
    checks = counterexamples()
    strategies = ["heuristic", "random"] + (["learned"] if args.checkpoint else [])
    results = {"counterexamples": checks, "cases": [], "summary": {}, "solve_seconds_total": 0.,
               "claim": "Geometry-only development probe; learning value pending trained control"}
    if not args.counterexamples_only:
        cases = load_frozen_cases("dev", limit=args.limit)
        for case in cases:
            row = {"case_id": case.case_id, "source": case.source, "input_summary": case.input_summary(), "variants": {}}
            for strategy in strategies:
                remaining = args.total_seconds - results["solve_seconds_total"]
                if remaining < .1:
                    record = {"error": "total inference resource stop", "solve_seconds": 0.,
                              "metrics": {"cost": 10., "is_feasible": False, "hpwl_gap": None}}
                else:
                    solver = Optimizer(strategy, seconds=min(args.seconds, remaining), seed=args.seed, checkpoint=args.checkpoint)
                    solve_start = time.perf_counter()
                    final = solver.solve(*case.solver_inputs)
                    elapsed = time.perf_counter() - solve_start
                    results["solve_seconds_total"] += elapsed
                    stages = {name: evaluate_case(case, positions, elapsed) for name, positions in solver.stages.items()}
                    # Final geometry stored once here; stage dict avoids a second
                    # copy of the same final coordinates.
                    record = stages.pop("final")
                    record.update(stages=stages, solve_seconds=elapsed, diagnostics=solver.last_stats,
                                  correction_cost_change=record["metrics"]["cost"] - stages["raw"]["metrics"]["cost"])
                row["variants"][strategy] = record
                print(json.dumps({"case": case.case_id, "strategy": strategy, "seconds": record["solve_seconds"],
                                  "feasible": record["metrics"]["is_feasible"], "Q": record["metrics"]["cost"]}), flush=True)
            results["cases"].append(row)
            write(args.output / "results.json", results)
        results["summary"] = summarize_variants(results["cases"], strategies)
        for strategy in strategies:
            diagnostics = [r["variants"][strategy].get("diagnostics", {}) for r in results["cases"]]
            results["summary"][strategy]["fallback_count"] = sum(d.get("fallback", False) for d in diagnostics)
    results["wall_seconds"] = time.perf_counter() - started
    write(args.output / "results.json", results)
    print(json.dumps(results["summary"] or checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
