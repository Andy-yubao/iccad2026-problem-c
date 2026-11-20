"""One authorized semantic-cache -> formal-train process, with no retries."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from datasets.paths import ROOT
from ..file_io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    launch = args.launch.resolve()
    auth = json.loads((launch / "authorization.json").read_text(encoding="utf-8"))
    if not auth["background_training_authorized"]:
        raise PermissionError("Explicit background authorization required")
    cache, run, gallery = [ROOT / auth[k] for k in ("cache", "run", "gallery")]

    def state(status, **extra):
        atomic_json(launch / "state.json", {"status": status, "pid": os.getpid(),
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run": str(run), "cache": str(cache), "gallery": str(gallery), "test_read": False, **extra})

    for phase, command in (
        ("prepare", [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.semantic.prepare", "--out", str(cache)]),
        ("train", [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.semantic.train",
            "--plan", str(launch / "plan.json"), "--run", str(run), "--cache", str(cache), "--gallery", str(gallery)])):
        if (launch / "STOP").exists() or (run / "STOP").exists():
            state("user_stopped")
            return
        if phase == "train":
            wait_start = time.monotonic()
            while not (launch / "TRAINER_READY").exists():
                if (launch / "STOP").exists() or (run / "STOP").exists():
                    state("user_stopped")
                    return
                if time.monotonic() - wait_start > 1200:
                    state("failed", reason="trainer_not_finalized_in_20_minutes")
                    return
                time.sleep(1)
        with (launch / f"{phase}.stdout.log").open("x", encoding="utf-8") as output, (launch / f"{phase}.stderr.log").open("x", encoding="utf-8") as errors:
            child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            state("preparing_images" if phase == "prepare" else "training", phase=phase, child_pid=child.pid, command=command)
            code = child.wait()
        if code:
            state("failed", phase=phase, exit_code=code)
            return
    result = json.loads((run / "state.json").read_text(encoding="utf-8"))
    state("finished", training_stop=result["status"], steps=result["steps"], epoch=result["epoch"])


if __name__ == "__main__":
    main()
