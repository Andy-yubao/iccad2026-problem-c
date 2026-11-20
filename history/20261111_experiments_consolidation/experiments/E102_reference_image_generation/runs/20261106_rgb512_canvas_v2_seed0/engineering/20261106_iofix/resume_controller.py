"""Single user-requested repair resume; preserve original failed launch logs."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path.cwd()
    run = root / "experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0"
    repair = run / "engineering/20261106_iofix"
    launch = root / "experiments/E102_reference_image_generation/runs/20261106_rgb512_canvas_v2_seed0_launch"
    gallery = root / "visualization/E102_reference_image_generation/20261106_rgb512_canvas_v2_seed0"
    command = [sys.executable, "-u", "-m", "experiments.E102_reference_image_generation.train",
        "--plan", str(launch / "plan.json"), "--cache", str(root / "experiments/E102_reference_image_generation/runs/prepared_rgb512_canvas_v2_20261106"),
        "--run", str(run), "--resume", "--io-migration", str(repair / "migration.json"), "--gallery", str(gallery)]

    def record(status, **values):
        state = {"status": status, "pid": os.getpid(), "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run": str(run), "gallery": str(gallery), "phase": "train_iofix_resume", "repair": str(repair),
            "stdout": str(launch / "train_resume_iofix.stdout.log"), "stderr": str(launch / "train_resume_iofix.stderr.log"),
            "test_read": False, **values}
        for folder in (repair, launch):
            temporary = folder / "state.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(folder / "state.json")

    if (run / "STOP").exists() or (launch / "STOP").exists():
        record("user_stopped")
        return
    with (launch / "train_resume_iofix.stdout.log").open("x", encoding="utf-8") as output, (launch / "train_resume_iofix.stderr.log").open("x", encoding="utf-8") as errors:
        child = subprocess.Popen(command, cwd=root, stdout=output, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record("training", child_pid=child.pid, command=command)
        code = child.wait()
    if code:
        record("failed", exit_code=code)
    else:
        state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        record("finished", training_stop=state["status"], epochs=state["epoch"], steps=state["steps"])


if __name__ == "__main__":
    main()
