"""Bounded, reproducible five-case research comparison (never opens local_test).

Example: python experiments/baseline/run.py --strategies row pack search no_soft
         --seconds 15 --seed 0 --output experiments/baseline/runs/new_dev5

Optimizers expose Optimizer(strategy=..., seconds=..., seed=...), with the
official seven-argument solve interface returning XYWH positions.  A subprocess
receives only permitted inputs; labels and baseline metrics stay with scoring.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import uuid
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HARD_WALL_LIMIT_SECONDS = 1140.0  # includes data loading, imports, workers and scoring
WORKER_STARTUP_GRACE_SECONDS = 30.0


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def solver_worker(connection, module_name, strategy, seconds, seed, threads, safe_inputs):
    """The child is passed no labels, reference metrics, or ResearchCase object."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            sys.path.insert(0, str(ROOT))
            import numpy as np
            import torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.set_num_threads(threads)
            module = importlib.import_module(module_name)
            solver = module.Optimizer(strategy=strategy, seconds=seconds, seed=seed)
            started = time.perf_counter()
            positions = solver.solve(*safe_inputs)
            solve_seconds = time.perf_counter() - started
            if hasattr(positions, "tolist"):
                positions = positions.tolist()
            payload = {"positions": positions, "solve_seconds": solve_seconds,
                       "worker_peak_memory_bytes": peak_worker_memory_bytes()}
            for field in ("last_stats", "last_diagnostics"):
                value = getattr(solver, field, None)
                if value is not None:
                    payload[field] = value
            payload["worker_log"] = buffer.getvalue()[-65536:]
        connection.send(payload)
    except BaseException:
        connection.send({"error": traceback.format_exc(), "worker_log": buffer.getvalue()[-65536:],
                         "worker_peak_memory_bytes": peak_worker_memory_bytes()})
    finally:
        connection.close()


def peak_worker_memory_bytes():
    """Process high-water mark includes interpreter/imports; no extra dependency."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.argtypes = []
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
            return None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, AttributeError):
        return None


def serialize_safe_inputs(inputs):
    """Serialize exactly the seven legal solve arguments, including empty shapes."""
    return [int(inputs[0]), *[
        {"values": value.tolist(), "shape": list(value.shape), "dtype": str(value.dtype).split(".")[-1]}
        for value in inputs[1:]]]


class JsonResultSink:
    def __init__(self, path):
        self.path = path

    def send(self, payload):
        write_json(self.path, payload)

    def close(self):
        pass


def file_worker_main():
    parser = argparse.ArgumentParser(description="Internal safe-input file worker")
    parser.add_argument("--worker-input", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.worker_input.read_text(encoding="utf-8"))
    import torch
    allowed_dtypes = {name: getattr(torch, name) for name in
                      ("float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool")}
    values = request["inputs"]
    if len(values) != 7:
        raise ValueError("Expected exactly seven legal solve arguments")
    safe_inputs = (int(values[0]), *[
        torch.tensor(value["values"], dtype=allowed_dtypes[value["dtype"]]).reshape(value["shape"])
        for value in values[1:]])
    solver_worker(JsonResultSink(args.worker_output), request["solver"], request["strategy"],
                  request["seconds"], request["seed"], request["threads"], safe_inputs)


def run_solver(case, module, strategy, seconds, seed, threads, deadline):
    started = time.perf_counter()
    timeout = min(seconds + WORKER_STARTUP_GRACE_SECONDS, deadline - started - 5)
    if timeout <= 0:
        return {"error": "Global wall budget exhausted before this call", "worker_wall_seconds": 0.0}
    cache_root = (ROOT / ".cache/research_workers").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    call_id = uuid.uuid4().hex
    input_path = cache_root / f"{call_id}.input.json"
    output_path = cache_root / f"{call_id}.output.json"
    process = None
    try:
        # Plain JSON copies the inputs; child mutations cannot affect scoring.
        write_json(input_path, {"solver": module, "strategy": strategy, "seconds": seconds,
                               "seed": seed, "threads": threads,
                               "inputs": serialize_safe_inputs(case.solver_inputs)})
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--worker-input", str(input_path), "--worker-output", str(output_path)]
        process = subprocess.Popen(command, cwd=str(ROOT), shell=False,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=max(.001, timeout - (time.perf_counter() - started)))
        except subprocess.TimeoutExpired:
            result = {"error": f"Optimizer worker exceeded {timeout:.3f}s hard timeout",
                      "resource_stop": "worker_timeout"}
        else:
            if output_path.is_file():
                try:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = {"error": "Optimizer worker returned invalid JSON: " + traceback.format_exc()}
            else:
                result = {"error": f"Optimizer worker exited without a result (exit={process.returncode})"}
    except OSError:
        result = {"error": "Failed to launch optimizer worker: " + traceback.format_exc()}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        # Remove only this invocation's two uniquely named files. A shared
        # cache directory avoids Windows restrictions on deleting directories.
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
    result["worker_wall_seconds"] = time.perf_counter() - started
    return result


def failure_result(error, elapsed=0.0, **details):
    return {"error": error, "metrics": {"is_feasible": False, "cost": 10.0,
            "hpwl_gap": None, "area_gap": None, "boundary_violations": None,
            "grouping_violations": None, "mib_violations": None, "runtime_seconds": elapsed}, **details}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", default="experiments.baseline.optimizer", help="Module exposing Optimizer")
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument("--strategy", help="Single strategy (default: search)")
    strategy_group.add_argument("--strategies", nargs="+", help="Compare strategies on the same five cases")
    parser.add_argument("--panel", choices=("dev", "public"), default="dev")
    parser.add_argument("--seconds", type=float, default=15, help="Per-case, per-strategy solver budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    args = parser.parse_args()
    strategies = args.strategies or [args.strategy or "search"]
    if len(set(strategies)) != len(strategies):
        parser.error("Strategy names must be unique")
    if args.threads < 1 or not math_is_positive_finite(args.seconds):
        parser.error("threads must be positive and seconds must be finite and > 0")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    call_count = 5 * len(strategies)
    planned_solver_seconds = call_count * args.seconds
    if planned_solver_seconds > 600:
        parser.error("This strategy stage is preregistered for at most 600 solver seconds in total; reduce strategies or seconds")
    # Include conservative startup/scoring overhead rather than splitting a long run.
    worst_case_seconds = call_count * (args.seconds + WORKER_STARTUP_GRACE_SECONDS + 4) + 60
    if worst_case_seconds >= HARD_WALL_LIMIT_SECONDS:
        parser.error(f"Requested run cannot be safely bounded below 20 minutes: estimate {worst_case_seconds:.0f}s; reduce strategies or seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    sys.path.insert(0, str(ROOT))
    started = time.perf_counter()
    deadline = started + HARD_WALL_LIMIT_SECONDS
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            import torch
            from datasets.cases import ROOT as eval_root, load_frozen_cases
            from scoring.evaluation import evaluate_case, summarize_variants
            from datasets.paths import resolved_data_paths
            torch.set_num_threads(args.threads)
            module = importlib.import_module(args.solver)
            if not hasattr(module, "Optimizer"):
                raise ValueError(f"{args.solver} must expose Optimizer")
            source_files = [*sorted(Path(__file__).resolve().parent.glob("*.py")),
                            *sorted((ROOT / "datasets").glob("*.py")),
                            *sorted((ROOT / "scoring").glob("*.py")), Path(module.__file__),
                            ROOT / "protocol/paths.json",
                            ROOT / "third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py"]
            source_hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                             hashlib.sha256(path.read_bytes()).hexdigest() for path in dict.fromkeys(source_files)}
            packages = {}
            for package in ("torch", "numpy", "shapely", "matplotlib", "scipy"):
                try:
                    packages[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    packages[package] = None
            config = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command_argv": [sys.executable, *sys.argv],
                "command_display": subprocess_command([sys.executable, *sys.argv]),
                "workspace": str(eval_root), "solver_module": args.solver, "strategies": strategies,
                "resolved_data_paths": resolved_data_paths(),
                "panel": args.panel, "seed": args.seed, "seconds_per_case_strategy": args.seconds,
                "planned_solver_seconds": planned_solver_seconds,
                "preregistered_research_wall_budget_seconds": 600,
                "conservative_wall_estimate_seconds": worst_case_seconds,
                "hard_wall_limit_seconds": HARD_WALL_LIMIT_SECONDS,
                "worker_startup_grace_seconds": WORKER_STARTUP_GRACE_SECONDS,
                "worker_transport": "same-interpreter subprocess with workspace temporary safe-input/output JSON; standard streams DEVNULL; no named pipes",
                "python_executable": sys.executable, "python_version": sys.version,
                "platform": platform.platform(), "packages": packages, "torch_cpu_threads": args.threads,
                "torch_load": "map_location=cpu, weights_only=False for the trusted archived official tensor files",
                "source_sha256": source_hashes,
                "split_file_sha256": hashlib.sha256((ROOT / "protocol/split_v1.json").read_bytes()).hexdigest(),
                "upstream_commit": json.loads((ROOT / "third_party/upstream_manifest.json").read_text())["commit"],
                "runtime_scoring": "RuntimeFactor=1.0; neutral local quality cost, not an official contest score",
                "memory_measurement": "per-worker process peak working set via Windows GetProcessMemoryInfo; Unix resource.ru_maxrss fallback; includes imports; null on unsupported or timed-out workers",
                "data_boundary": "five frozen dev samples or five explicit public compatibility cases only; local_test sealed",
                "solver_input_boundary": "ordinary inputs and masked fixed/preplaced specifications; no free-block labels or reference metrics",
            }
            write_json(output / "config.json", config)
            cases = load_frozen_cases(args.panel)
            results = {"status": "running", "panel": args.panel, "strategies": strategies,
                       "runtime_factor": 1.0, "official_contest_score": None,
                       "score_note": config["runtime_scoring"], "cases": []}
            for case in cases:
                case_record = {"case_id": case.case_id, "source": case.source,
                               "input_summary": case.input_summary(),
                               "block_constraints": case.solver_inputs[5].tolist(), "variants": {}}
                results["cases"].append(case_record)
                for strategy in strategies:
                    raw = run_solver(case, args.solver, strategy, args.seconds, args.seed, args.threads, deadline)
                    worker_log = raw.pop("worker_log", "")
                    if worker_log:
                        print(f"[{case.case_id} / {strategy}] worker log:\n{worker_log}", flush=True)
                    elapsed = raw.get("solve_seconds", raw["worker_wall_seconds"])
                    if "error" in raw:
                        error = raw.pop("error")
                        record = failure_result(error, elapsed, **raw)
                    else:
                        positions = raw.pop("positions")
                        try:
                            record = {**evaluate_case(case, positions, elapsed), **raw}
                        except Exception:
                            record = failure_result(traceback.format_exc(), elapsed, **raw)
                    case_record["variants"][strategy] = record
                    metrics = record["metrics"]
                    print(f"{case.case_id} strategy={strategy} n={case.block_count} feasible={metrics['is_feasible']} neutral_cost={metrics['cost']:.6f} solve={elapsed:.3f}s", flush=True)
                    if "error" in record:
                        print(record["error"], flush=True)
                    # Preserve completed work if the user interrupts or a later call fails.
                    results["wall_seconds"] = time.perf_counter() - started
                    write_json(output / "results.json", results)
            results["summary"] = summarize_variants(results["cases"], strategies)
            results["wall_seconds"] = time.perf_counter() - started
            results["status"] = "completed_with_errors" if any(
                "error" in variant for case in results["cases"] for variant in case["variants"].values()) else "completed"
            write_json(output / "results.json", results)
            print(json.dumps(results["summary"], ensure_ascii=False, indent=2), flush=True)
            print(f"Saved: {output / 'results.json'}", flush=True)


def math_is_positive_finite(value):
    import math
    return math.isfinite(value) and value > 0


def subprocess_command(argv):
    if os.name == "nt":
        import subprocess
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker-input"]:
        file_worker_main()
    else:
        main()
