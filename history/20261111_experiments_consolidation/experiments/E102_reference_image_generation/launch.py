"""One authorized prepare -> train chain; no retries or other experiments."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from datasets.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    folder = args.launch.resolve()
    config = json.loads((folder / "authorization.json").read_text(encoding="utf-8"))
    if not config["background_training_authorized"]:
        raise PermissionError("Explicit background launch authorization required")
    plan = folder / "plan.json"
    run, cache, gallery = [ROOT / config[key] for key in ("run", "cache", "gallery")]

    def state(status, **values):
        record = {"status": status, "pid": os.getpid(), "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run": str(run), "cache": str(cache), "gallery": str(gallery), "test_read": False, **values}
        temporary = folder / "state.json.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(folder / "state.json")

    state("preparing_images")
    for phase, command in (
        ("prepare", [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.prepare",
                     "--plan", str(plan), "--out", str(cache)]),
        ("train", [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.train",
                   "--plan", str(plan), "--cache", str(cache), "--run", str(run), "--gallery", str(gallery)])):
        if (folder / "STOP").exists() or (run / "STOP").exists():
            state("user_stopped", next_phase=phase)
            return
        with (folder / f"{phase}.stdout.log").open("w", encoding="utf-8") as output, (folder / f"{phase}.stderr.log").open("w", encoding="utf-8") as errors:
            child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            state("preparing_images" if phase == "prepare" else "training", child_pid=child.pid, phase=phase, command=command)
            code = child.wait()
        if code:
            state("failed", phase=phase, exit_code=code)
            return
    result = json.loads((run / "state.json").read_text(encoding="utf-8"))
    state("finished", training_stop=result["status"], epochs=result["epoch"], steps=result["steps"])


if __name__ == "__main__":
    main()
