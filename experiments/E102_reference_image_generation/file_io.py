"""Bounded Windows sharing-conflict retries; no optimizer retries."""
import json
from pathlib import Path
import sys
import time


def replace_file(source, target, *, timeout=2.):
    deadline = time.monotonic() + timeout
    delay = .01
    while True:
        try:
            Path(source).replace(target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, max(0., deadline - time.monotonic())))
            delay = min(delay * 2, .2)


def atomic_json(path, value, *, timeout=2.):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    replace_file(temporary, path, timeout=timeout)


def advisory_append(path, value):
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")
        return True
    except PermissionError as error:
        if getattr(error, "winerror", None) not in (5, 32, 33):
            raise
        print(json.dumps({"event": "log_append_sharing_conflict", "path": str(path),
            "winerror": error.winerror, "training_continues": True}), file=sys.stderr, flush=True)
        return False


def advisory_json(path, value):
    """A failed status/heartbeat publication must not discard a valid update."""
    try:
        atomic_json(path, value, timeout=.5)
        return True
    except PermissionError as error:
        if getattr(error, "winerror", None) not in (5, 32, 33):
            raise
        warning = {"event": "status_write_sharing_conflict", "path": str(path),
            "steps": value.get("steps"), "winerror": error.winerror, "training_continues": True}
        print(json.dumps(warning), file=sys.stderr, flush=True)
        advisory_append(path.parent / "io_warnings.jsonl", warning)
        return False
