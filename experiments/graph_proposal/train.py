"""User-run E1 training, with batch-level budget checks and strict resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import zipfile
import uuid
import math

PROCESS_STARTED = time.perf_counter()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from datasets.paths import ROOT, resolved_data_paths
from .data import MEMBERS_PATH, TrainMembers, digest, member_manifest
from .model import GraphProposal

HERE = Path(__file__).resolve().parent


def source_hashes():
    sources = [HERE / name for name in ("data.py", "model.py", "train.py")]
    sources += list((ROOT / "datasets").glob("*.py")) + list((ROOT / "scoring").glob("*.py"))
    sources += list((ROOT / "third_party/FloorSet/iccad2026contest").glob("*.py"))
    return {str(path.relative_to(ROOT).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)}


def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def git_status():
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else "unavailable"


def source_snapshot(path):
    sources = [HERE / name for name in ("data.py", "model.py", "train.py")]
    sources += list((ROOT / "datasets").glob("*.py")) + list((ROOT / "scoring").glob("*.py"))
    sources += list((ROOT / "third_party/FloorSet/iccad2026contest").glob("*.py"))
    sources += [ROOT / "protocol/split_v1.json", ROOT / "protocol/paths.json", MEMBERS_PATH]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT).as_posix())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_with_retry(temporary, path):
    """Windows readers may briefly hold a destination without delete sharing."""
    deadline = time.perf_counter() + 3.0
    delay = .01
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.perf_counter() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, .2)


def save_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def recorded_seconds(run_dir):
    """Recover the highest durable time, including old failed-run evidence."""
    seconds = 0.0
    for name in ('budget_ledger.json', 'error.json', 'results.json'):
        path = run_dir / name
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            seconds = max(seconds, float(value.get('accumulated_seconds', 0)))
    journal = run_dir / 'budget_ledger.jsonl'
    if journal.exists():
        lines = journal.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines)-1: break  # interrupted final append
                raise
            seconds = max(seconds, float(value['accumulated_seconds']))
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid budget evidence')
    return seconds


def append_budget(run_dir, value):
    """An append-only journal avoids per-batch replacement of an indexed file."""
    path = run_dir / 'budget_ledger.jsonl'
    with path.open('a+', encoding='utf-8', newline='\n') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(stream.tell()-1)
            last = stream.read()
            if last != '\n':
                # Retain interrupted bytes separately; do not append onto invalid JSON.
                raise ValueError('Budget journal has an incomplete tail; inspect before resuming')
        stream.write(json.dumps(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("deterministic", "flow"), default="deterministic")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare-members-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--legacy-protocol',action='store_true',help='Explicit historical reproduction only; new training uses train_supervised.py')
    parser.add_argument("--previous-attempt", type=Path,
                        help="Restart without weights in a new directory, charging this failed attempt's recorded time")
    parser.add_argument("--budget-seconds", type=float, default=3600,
                        help="Total accumulated wall budget, including resumed invocations; at most 3600")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional short smoke cap per invocation")
    args = parser.parse_args()
    if not args.legacy_protocol:
        parser.error('Historical trainer disabled by default. New deterministic training: python -B -m experiments.graph_proposal.train_supervised --run-dir <new_run>. Use --legacy-protocol only to reproduce old records.')
    if not 0 < args.budget_seconds <= 3600 or args.batch_size < 1 or args.threads < 1:
        parser.error("budget must be (0,3600], batch size and threads must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be positive")
    if args.resume and args.previous_attempt:
        parser.error('--resume and --previous-attempt are mutually exclusive')
    started = PROCESS_STARTED
    manifest = member_manifest(create=args.prepare_members_only)
    if args.prepare_members_only:
        print(json.dumps({"manifest": str(MEMBERS_PATH), "members_sha256": manifest["members_sha256"],
                          "sample_count": manifest["sample_count"], "tensor_reads": 0}))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required for training")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((HERE / "runs").resolve()):
        parser.error("run-dir must be inside experiments/graph_proposal/runs")
    if run_dir.exists() and not args.resume:
        parser.error("run-dir already exists; choose a new name or explicitly --resume")
    if args.resume and not (run_dir / "checkpoints/latest.pt").exists():
        parser.error("resume checkpoint missing")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    # Fail before a new run is created if local data paths are unavailable.
    paths = resolved_data_paths()
    members = TrainMembers(manifest)
    data_stats = [{"shard": entry["shard"], "bytes": (members.root / entry["shard"]).stat().st_size,
                   "mtime_ns": (members.root / entry["shard"]).stat().st_mtime_ns}
                  for entry in manifest["selected_shards"]]
    identity = dict(source_sha256=source_hashes(), members_sha256=manifest["members_sha256"],
                    split_sha256=manifest["split_sha256"], resolved_data_paths=paths,
                    data_stat_sha256=digest(data_stats), seed=args.seed, batch_size=args.batch_size,
                    kind=args.kind, device=str(device), threads=args.threads,
                    torch_version=torch.__version__)
    model = GraphProposal(args.kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epoch, cursor, total_steps, samples_seen, prior_seconds = 0, 0, 0, 0, 0.0
    invocation_history = []
    previous_attempt = None
    if args.previous_attempt:
        previous_attempt = args.previous_attempt.resolve()
        if not previous_attempt.is_relative_to((HERE / 'runs').resolve()) or previous_attempt == run_dir:
            parser.error('previous-attempt must be a different graph_proposal run')
        previous_config = json.loads((previous_attempt / 'config.json').read_text(encoding='utf-8'))
        old_identity = previous_config['identity']
        if any(old_identity.get(key) != identity[key] for key in identity if key != 'source_sha256'):
            raise ValueError('Restart changed data, model kind, seed, batch, device or environment')
        if (previous_attempt / 'checkpoints/latest.pt').exists():
            raise ValueError('Previous attempt has weights; review recovery instead of silently restarting')
        prior_seconds = math.ceil(recorded_seconds(previous_attempt))
    if args.resume:
        checkpoint = torch.load(run_dir / "checkpoints/latest.pt", map_location=device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume identity mismatch: source, members, split, data location/stat, seed, batch, device, threads and torch must match")
        if checkpoint["model_config"] != model.config:
            raise ValueError("Model configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["rng_python"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if checkpoint["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
        epoch, cursor = checkpoint["epoch"], checkpoint["cursor"]
        total_steps, samples_seen = checkpoint["total_steps"], checkpoint["samples_seen"]
        prior_seconds = checkpoint["accumulated_seconds"]
        prior_seconds = max(prior_seconds, recorded_seconds(run_dir))
        invocation_history = checkpoint["invocations"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    snapshot = run_dir / "source_snapshot.zip"
    if not args.resume:
        source_snapshot(snapshot)
    invocation = dict(command=[sys.executable, *sys.argv], requested_budget_seconds=args.budget_seconds,
                      prior_accumulated_seconds=prior_seconds, max_batches=args.max_batches,
                      previous_attempt=str(previous_attempt) if previous_attempt else None,
                      restart_weights='initial_random_seed' if previous_attempt else None)
    invocation_history.append(invocation)
    save_json(run_dir / "config.json", dict(identity=identity, model=model.config, optimizer={"name": "AdamW", "lr": 1e-3,
              "weight_decay": 1e-4}, manifest=str(MEMBERS_PATH.relative_to(ROOT)), sample_count=manifest["sample_count"],
              python=platform.python_version(), git_revision=git_revision(), git_status=git_status(),
              source_snapshot={"path": snapshot.name, "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()},
              device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
              data_identity_method="selected shard path/size/mtime_ns; no repeated full dataset content hash",
              invocations=invocation_history))
    stop_requested = [False]
    def request_stop(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    def elapsed():
        return prior_seconds + time.perf_counter() - started

    def save_checkpoint(reason):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        payload = dict(identity=identity, model_config=model.config, model=model.state_dict(),
                       optimizer=optimizer.state_dict(), rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       epoch=epoch, cursor=cursor, total_steps=total_steps, samples_seen=samples_seen,
                       accumulated_seconds=elapsed(), invocations=invocation_history, stop_reason=reason)
        temporary = run_dir / ("checkpoints/latest." + uuid.uuid4().hex + ".pt.tmp")
        torch.save(payload, temporary)
        replace_with_retry(temporary, run_dir / "checkpoints/latest.pt")
        append_budget(run_dir, {"accumulated_seconds": elapsed(), "completed_steps": total_steps})

    model.train()
    batches_this_invocation = 0
    last_batch_seconds = 0.0
    last_save = time.perf_counter()
    last_loss = None
    reason = "resource_budget"
    try:
        save_checkpoint('initialized' if not args.resume else 'resumed')
        while True:
            # Reserve a checkpoint margin and one observed batch. No long epoch starts unchecked.
            if elapsed() + max(2.0, 1.5 * last_batch_seconds) >= args.budget_seconds:
                reason = "resource_budget"
                break
            if stop_requested[0]:
                reason = "user_stop_after_batch"
                break
            if args.max_batches is not None and batches_this_invocation >= args.max_batches:
                reason = "smoke_batch_cap"
                break
            order = members.epoch_order(epoch, args.seed)
            indices = order[cursor:cursor + args.batch_size]
            if not indices:
                epoch, cursor = epoch + 1, 0
                continue
            batch_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for index in indices:
                inputs, reference = members[index]
                loss = model.training_loss(inputs, reference) / len(indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                batch_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            last_batch_seconds = time.perf_counter() - batch_start
            cursor += len(indices)
            total_steps += 1
            samples_seen += len(indices)
            batches_this_invocation += 1
            last_loss = batch_loss
            append_budget(run_dir, {"accumulated_seconds": elapsed(), "completed_steps": total_steps})
            if batches_this_invocation == 1 or total_steps % 10 == 0:
                entry = dict(step=total_steps, samples_seen=samples_seen, epoch=epoch, cursor=cursor,
                             loss=batch_loss, accumulated_seconds=elapsed(), batch_seconds=last_batch_seconds)
                with (run_dir / "run.log").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
            if batches_this_invocation == 1 or time.perf_counter() - last_save >= 60:
                save_checkpoint("periodic")
                last_save = time.perf_counter()
    except Exception:
        # The preceding completed-batch checkpoint remains the safe resume point.
        # Do not serialize a potentially partial optimizer update as resumable.
        try:
            save_json(run_dir / "error.json", {"message": "Training exception; resume only the last completed checkpoint",
                                               "completed_steps_in_memory": total_steps, "accumulated_seconds": elapsed()})
            append_budget(run_dir, {"accumulated_seconds": elapsed(), "completed_steps": total_steps})
        except Exception as logging_error:
            print(f'Could not persist exception evidence: {logging_error}', file=sys.stderr)
        raise
    save_checkpoint(reason)
    save_json(run_dir / "results.json", dict(status="stopped", stop_reason=reason, converged=False,
              total_steps=total_steps, samples_seen=samples_seen, epoch=epoch, cursor=cursor, last_loss=last_loss,
              accumulated_seconds=elapsed(), requested_budget_seconds=args.budget_seconds,
              checkpoint="checkpoints/latest.pt", members_sha256=manifest["members_sha256"],
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))
    print(json.dumps({"stop_reason": reason, "steps": total_steps, "accumulated_seconds": elapsed()}))


if __name__ == "__main__":
    main()
