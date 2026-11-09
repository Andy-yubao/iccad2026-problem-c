"""Single authorized continuation, preserving stopped parents and frozen trainer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.E102_reference_image_generation.file_io import atomic_json

LAUNCH = Path(__file__).resolve().parent
PARENT = LAUNCH.parent / "20261106_semantic_v21_bf16_b4_seed0"
RUN = LAUNCH.parent / "20261106_semantic_v21_bf16_b4_resume5_seed0"
CACHE = LAUNCH.parent / "prepared_semantic_v21_20261106"
GALLERY = ROOT / "visualization/E102_reference_image_generation/20261106_semantic_v21_bf16_b4_seed0"
FP32 = LAUNCH.parent / "20261106_semantic_v21_seed0"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def status(value, **extra):
    atomic_json(LAUNCH / "state.json", {"status": value, "pid": os.getpid(),
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run": str(RUN), "parent_run": str(PARENT), "gallery": str(GALLERY),
        "test_read": False, **extra})


def main():
    auth = read(LAUNCH / "authorization.json")
    if not auth["background_training_authorized"]:
        raise PermissionError("Explicit authorization required")
    if (LAUNCH / "STOP").exists():
        status("user_stopped")
        return
    identity = read(PARENT / "identity.json")
    for name, digest in identity["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed: " + name)
    for role, digest in identity["cache_metadata_sha256"].items():
        if hashlib.sha256((CACHE / role / "metadata.json").read_bytes()).hexdigest() != digest:
            raise ValueError("Cache identity changed: " + role)
    RUN.mkdir(exist_ok=False)
    copies = {}
    for name in ("latest.pt", "best.pt", "identity.json", "source_snapshot.zip", "plan.json",
                 "state.json", "progress.jsonl", "updates.jsonl", "migration_baseline.json"):
        source, target = PARENT / name, RUN / name
        shutil.copy2(source, target)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != before:
            raise ValueError("Copy mismatch: " + name)
        copies[name] = before
    shutil.copy2(PARENT / "plan.json", LAUNCH / "plan.json")
    atomic_json(RUN / "continuation.json", {"parent": str(PARENT), "copied_sha256": copies,
        "state_preserved": "Model, Adam, scheduler, RNG, corruption RNG, order/cursor, cumulative time and best/latest",
        "control_only_bound": "After completed epoch5 fixed monitor, request safe stop; original cumulative8h and normal early stop remain active",
        "parent_STOP_preserved": True, "no_precision_or_objective_change": True, "test_read": False})
    command = [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.semantic.train",
        "--plan", str(LAUNCH / "plan.json"), "--run", str(RUN), "--cache", str(CACHE),
        "--gallery", str(GALLERY), "--resume"]
    dashboard_command = [sys.executable, "-u", "-m", "visualization.live_training",
        "--run", str(RUN), "--parent", str(FP32), "--gallery", str(GALLERY)]
    dashboard = None
    stop_reason = None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (LAUNCH / "train.stdout.log").open("x", encoding="utf-8") as out, \
         (LAUNCH / "train.stderr.log").open("x", encoding="utf-8") as err, \
         (LAUNCH / "dashboard.stdout.log").open("x", encoding="utf-8") as dash_out, \
         (LAUNCH / "dashboard.stderr.log").open("x", encoding="utf-8") as dash_err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, creationflags=flags)
        status("resuming", child_pid=child.pid, command=command)
        while child.poll() is None:
            current = read(RUN / "state.json")
            if dashboard is None and current.get("status") == "running":
                dashboard = subprocess.Popen(dashboard_command, cwd=ROOT, stdout=dash_out,
                    stderr=dash_err, creationflags=flags)
                status("training", child_pid=child.pid, dashboard_pid=dashboard.pid, command=command)
            if (LAUNCH / "STOP").exists() and not (RUN / "STOP").exists():
                stop_reason = "user_stop"
                (RUN / "STOP").write_text("User requested safe stop through continuation launch.\n", encoding="utf-8")
            if current.get("epoch", 0) >= 5 and not (RUN / "STOP").exists():
                stop_reason = "authorized_approximately_five_pass_window_completed"
                atomic_json(LAUNCH / "boundary_stop.json", {"reason": stop_reason, "state": current,
                    "original_time_budget_seconds": 28800, "test_read": False})
                (RUN / "STOP").write_text(stop_reason + "\n", encoding="utf-8")
                status("stopping_at_authorized_bound", child_pid=child.pid, reason=stop_reason)
            time.sleep(1)
        code = child.wait()
        current = read(RUN / "state.json")
        status("failed" if code else "finished", exit_code=code,
            training_stop=current.get("status"), boundary_reason=stop_reason,
            steps=current.get("steps"), seen_count=current.get("seen_count"))
        if dashboard is not None:
            dashboard.wait(timeout=20)
        else:
            subprocess.run(dashboard_command + ["--once"], cwd=ROOT, stdout=dash_out, stderr=dash_err,
                creationflags=flags, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise
