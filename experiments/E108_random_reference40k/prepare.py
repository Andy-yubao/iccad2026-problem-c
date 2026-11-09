"""User-authorized, finite CPU dataset preparation. Never trains a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import secrets
import time
import traceback

from datasets.paths import ROOT, dataset_paths
from datasets.random_reference import REGISTRY, digest, registry

RUN = ROOT / "experiments/E108_random_reference40k/runs/20261106_random40k_v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def output_root():
    paths = json.loads((ROOT / "protocol/paths.json").read_text(encoding="utf-8"))
    path = (ROOT / paths["random_reference40k_root"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Output escapes workspace")
    return path


def prepare():
    if REGISTRY.exists():
        raise FileExistsError("Do not overwrite a frozen new dataset")
    root = dataset_paths()["training_root"]
    shards = sorted(p.relative_to(root).as_posix() for p in root.glob("worker_*/layouts_*.th"))
    if len(shards) != 9000:
        raise ValueError(f"Expected 9000 raw shards, got {len(shards)}")
    seed = secrets.randbits(128)
    rng = random.Random(seed)
    rng.shuffle(shards)
    partitions = dict(train=shards[:8100], dev=shards[8100:8550], test=shards[8550:])
    value = {"dataset_version": "random_reference40k.v1", "policy": "research.original_no_mib.v1",
             "created_at": now(), "seed": seed, "rows_per_shard": 112,
             "sampling": "Fresh random.Random seed; shuffle all 9000 lexically sorted raw shards, split 8100/450/450; uniform sample without replacement from each new shard pool; no legacy metadata or payload used for selection.",
             "historical_exposure": "Unrestricted resampling may overlap old exposed members; not unseen for historical models.",
             "test_authorization": "Preparation/rendering only; no model evaluation or test tuning.",
             "render": {"resolution": 512, "public_canvas": "max(2*sqrt(sum(original_areas)), max_public_pin_xy+0.05*sqrt(sum(original_areas)), max_preplaced_extent+0.05*sqrt(sum(original_areas)))", "colors": "visualization.raster_layouts.module_palette", "antialias": "exact subpixel coverage", "no_crop_no_repair": True},
             "shard_partitions": partitions, "sets": {}, "panels": {}}
    for role, count in (("train", 36000), ("dev", 2000), ("test", 2000)):
        picks = rng.sample(range(len(partitions[role]) * 112), count)
        records = []
        for index, pick in enumerate(picks):
            records.append({"id": f"{role}_{index:05d}", "member": [partitions[role][pick // 112], pick % 112],
                            "image": f"{role}/images/{index:05d}.png",
                            "payload": f"{role}/samples/{index:05d}.npz"})
        value["sets"][role] = {"count": count, "records": records}
    for role in ("train", "dev"):
        value["panels"][role + "_monitor500"] = rng.sample(range(value["sets"][role]["count"]), 500)
    value["manifest_sha256"] = digest(value)
    write_json(REGISTRY, value)
    registry()
    protected = [ROOT / "protocol" / name for name in
                 ("split_v1.json", "workset_v2.json", "derived_workset_v3.json", "reference_workset_v5.json", "reference_workset_v6.json")]
    authorization = {"created_at": now(), "user_authorized": "Change future MIB policy and prepare fresh random 40000 images: 36000 train / 2000 dev / 2000 test; start preparation in background.",
                     "training_authorized": False, "test_model_evaluation_authorized": False,
                     "test_preparation_authorized": True, "new_split_overrides_legacy_for_this_dataset_only": True,
                     "manifest_sha256": value["manifest_sha256"], "workers": 4,
                     "legacy_hashes_before": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
                     "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                       (Path(__file__), ROOT / "datasets/random_reference.py", ROOT / "visualization/raster_layouts.py", ROOT / "scoring/research_no_mib.py", ROOT / "protocol/research_no_mib_v1.json")}}
    write_json(RUN / "authorization.json", authorization)
    write_json(RUN / "state.json", {"status": "prepared_not_started", "updated_at": now(), "manifest_sha256": value["manifest_sha256"]})
    print(json.dumps({"counts": {k: v["count"] for k, v in value["sets"].items()}, "manifest_sha256": value["manifest_sha256"]}), flush=True)


def worker(task):
    import numpy as np
    import torch
    from datasets.cases import allowed_target_positions
    from visualization.raster_layouts import reference_rgb, save_rgb
    torch.set_num_threads(1)
    shard, records, destination = task
    source = (dataset_paths()["training_root"] / shard).resolve()
    if not source.is_relative_to(dataset_paths()["training_root"]):
        raise ValueError("Source path escapes raw root")
    contents = source.read_bytes()
    raw = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)
    if len(raw) != 7 or any(len(column) != 112 for column in raw):
        raise ValueError("Unexpected raw schema")
    done = []
    for role, record in records:
        row = record["member"][1]
        f = raw[0][row].detach().cpu().numpy().copy()
        rects = raw[5][row][:, [2, 3, 0, 1]].detach().cpu().numpy().copy()
        areas, constraints = f[:, 0].copy(), f[:, 1:].copy()
        n = len(areas)
        if f.shape != (n, 6) or rects.shape != (n, 4) or not np.isfinite(f).all() or not np.isfinite(rects).all() or (areas <= 0).any() or (rects[:, 2:] <= 0).any():
            raise ValueError(f"Malformed selected record {record['id']}; no skip/replacement")
        b2b, p2b, pins = [raw[i][row].detach().cpu().numpy().copy() for i in (1, 2, 3)]
        if not all(np.isfinite(x).all() for x in (b2b, p2b, pins)):
            raise ValueError("Nonfinite public connectivity")
        specs = allowed_target_positions(rects.tolist(), constraints).numpy()
        scale = float(np.sqrt(areas.astype(np.float64).sum()))
        pre = constraints[:, 1] != 0
        pre_extent = float((specs[pre, :2] + specs[pre, 2:]).max()) if pre.any() else 0.
        pin_extent = max(0., float(pins[:, :2].max())) if pins.size else 0.
        side = max(2 * scale, max(pin_extent, pre_extent) + .05 * scale)
        if (rects[:, :2] < 0).any() or (rects[:, :2] + rects[:, 2:] > side + 1e-6).any():
            raise ValueError(f"Reference outside public canvas: {record['id']}; stop, no crop/filter")
        image_path, payload_path = Path(destination) / record["image"], Path(destination) / record["payload"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists() or payload_path.exists():
            raise FileExistsError("No implicit overwrite/resume")
        save_rgb(image_path, reference_rgb(rects, side, 512))
        np.savez_compressed(payload_path, areas=areas, constraints=constraints, specs=specs,
                            b2b=b2b, p2b=p2b, pins=pins, reference_xywh=rects,
                            stored_metrics=raw[6][row].detach().cpu().numpy(),
                            canvas_side=np.float64(side))
        done.append({"id": record["id"], "role": role, "n": n, "canvas_side": side,
                     "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                     "payload_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest()})
    return {"shard": shard, "source_sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents), "records": done}


def tasks(value, destination, check=False):
    grouped = defaultdict(list)
    roles = ("train",) if check else ("train", "dev", "test")
    for role in roles:
        records = value["sets"][role]["records"][:5] if check else value["sets"][role]["records"]
        for record in records:
            grouped[record["member"][0]].append((role, record))
    return [(shard, records, str(destination)) for shard, records in sorted(grouped.items())]


def bounded_results(executor, task_list):
    """Bound queued work so failure/STOP cannot schedule the remaining corpus."""
    remaining = iter(task_list)
    pending = {executor.submit(worker, task) for task in [next(remaining) for _ in range(min(8, len(task_list)))]}
    try:
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
            for _ in finished:
                task = next(remaining, None)
                if task is not None:
                    pending.add(executor.submit(worker, task))
    finally:
        for future in pending:
            future.cancel()


def check():
    from PIL import Image
    value = registry()
    destination = RUN / "check5"
    outputs = [worker(task) for task in tasks(value, destination, check=True)]
    for record in value["sets"]["train"]["records"][:5]:
        with Image.open(destination / record["image"]) as img:
            img.verify()
        with Image.open(destination / record["image"]) as img:
            assert img.size == (512, 512) and img.mode == "RGB"
    write_json(RUN / "check5.json", {"status": "passed", "count": 5, "test_read": False, "outputs": outputs})
    print("Five selected train records: raw payload + RGB PNG passed", flush=True)


def run():
    value, destination = registry(), output_root()
    if not (RUN / "check5.json").exists():
        raise RuntimeError("Required five-case check missing")
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    counts = Counter()
    histogram = {role: Counter() for role in value["sets"]}
    state = {"status": "running", "pid": os.getpid(), "started_at": now(),
             "manifest_sha256": value["manifest_sha256"], "target": 40000, "workers": 4}
    task_list = tasks(value, destination)
    try:
        with (destination / "checksums.jsonl").open("x", encoding="utf-8") as ledger:
            with ProcessPoolExecutor(max_workers=4) as executor:
                for shard_index, result in enumerate(bounded_results(executor, task_list), 1):
                    ledger.write(json.dumps(result) + "\n")
                    for record in result["records"]:
                        counts[record["role"]] += 1
                        histogram[record["role"]][record["n"]] += 1
                    if shard_index % 20 == 0 or shard_index == len(task_list):
                        ledger.flush()
                        state.update(updated_at=now(), completed=dict(counts), completed_shards=shard_index,
                                     elapsed_seconds=time.monotonic() - start)
                        write_json(RUN / "state.json", state)
                        print(f"{sum(counts.values())}/40000 images; {shard_index}/{len(task_list)} shards", flush=True)
                    if (RUN / "STOP").exists():
                        raise InterruptedError("User STOP: preserve outputs; do not auto-resume")
        expected = {"train": 36000, "dev": 2000, "test": 2000}
        if dict(counts) != expected:
            raise ValueError("Completion count mismatch")
        for role, count in expected.items():
            if len(list((destination / role / "images").glob("*.png"))) != count or len(list((destination / role / "samples").glob("*.npz"))) != count:
                raise ValueError("Output file count mismatch")
        auth = json.loads((RUN / "authorization.json").read_text(encoding="utf-8"))
        for name, sha in auth["legacy_hashes_before"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Protected legacy index changed")
        report = {"status": "completed", "completed_at": now(), "counts": dict(counts),
                  "manifest_sha256": value["manifest_sha256"], "module_count_histograms": histogram,
                  "elapsed_seconds": time.monotonic() - start, "legacy_indexes_unchanged": True,
                  "test_preparation_only": True, "model_training_or_evaluation": False,
                  "checksums_sha256": hashlib.sha256((destination / "checksums.jsonl").read_bytes()).hexdigest()}
        write_json(destination / "report.json", report)
        write_json(RUN / "report.json", report)
        state.update(status="completed", updated_at=now(), completed=dict(counts), elapsed_seconds=report["elapsed_seconds"])
        write_json(RUN / "state.json", state)
    except BaseException as error:
        state.update(status="stopped" if isinstance(error, InterruptedError) else "failed",
                     updated_at=now(), completed=dict(counts), error=repr(error), traceback=traceback.format_exc())
        write_json(RUN / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "check", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "check": check, "run": run}[args.action]()
