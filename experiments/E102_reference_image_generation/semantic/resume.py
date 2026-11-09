"""One authorized BF16/batch4 continuation after its necessary four-case check."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from datasets.paths import ROOT
from ..file_io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    launch = args.launch.resolve()
    auth = json.loads((launch / "authorization.json").read_text(encoding="utf-8"))
    check = json.loads((launch / "engineering_check4/result.json").read_text(encoding="utf-8"))
    if not auth["background_training_authorized"] or check["status"] != "passed":
        raise PermissionError("Authorized migration and passed actual-update check required")
    run, cache, gallery, parent = [ROOT / auth[k] for k in ("run", "cache", "gallery", "parent_run")]
    command = [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.semantic.train",
        "--plan", str(launch / "plan.json"), "--cache", str(cache), "--run", str(run), "--gallery", str(gallery),
        "--migrate-from", str(parent), "--migration", str(launch / "migration.json")]

    def state(status, **extra):
        atomic_json(launch / "state.json", {"status": status, "pid": os.getpid(),
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "run": str(run),
            "parent_run": str(parent), "gallery": str(gallery), "test_read": False, **extra})

    if (run / "STOP").exists() or (launch / "STOP").exists():
        state("user_stopped")
        return
    with (launch / "train.stdout.log").open("x", encoding="utf-8") as output, (launch / "train.stderr.log").open("x", encoding="utf-8") as errors:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state("migration_and_training", child_pid=child.pid, command=command)
        code = child.wait()
    if code:
        state("failed", exit_code=code)
        return
    result = json.loads((run / "state.json").read_text(encoding="utf-8"))
    state("finished", training_stop=result["status"], steps=result["steps"], seen_count=result["seen_count"])


if __name__ == "__main__":
    main()
