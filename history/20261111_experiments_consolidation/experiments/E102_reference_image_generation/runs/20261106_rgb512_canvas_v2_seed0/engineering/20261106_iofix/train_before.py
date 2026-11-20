"""Explicit foreground image training, with full best/latest state."""
import argparse
import hashlib
import json
import random
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.paths import ROOT
from .model import ReferenceImageGenerator, image_loss
from .prepare import CachedImages


class TrainingStopped(Exception):
    pass


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loader(dataset, indices, config):
    return DataLoader(Subset(dataset, indices), batch_size=config["batch"], shuffle=False,
        num_workers=config["workers"], pin_memory=True,
        generator=torch.Generator().manual_seed(config["seed"]))


@torch.no_grad()
def monitor(model, dataset, config, device, should_stop):
    model.eval()
    total, count = 0., 0
    for batch in loader(dataset, list(range(len(dataset))), config):
        if reason := should_stop():
            raise TrainingStopped(reason)
        batch = move(batch, device)
        losses = image_loss(model(batch), batch["target"])
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite image monitoring loss")
        total += float(losses.sum())
        count += len(losses)
    return total / count


def snapshot(run):
    paths = list(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("plan.json")]
    paths += [ROOT / p for p in ("visualization/raster_layouts.py", "visualization/image_gallery.py", "datasets/worksets.py",
        "datasets/reference_layouts.py", "datasets/cases.py", "datasets/paths.py",
        "protocol/paths.json", "protocol/split_v1.json", "protocol/reference_workset_v5.json")]
    hashes = {}
    with zipfile.ZipFile(run / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, relative)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gallery", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    config = plan["training"]
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Registered CUDA training requires an available GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    train, train_monitor, dev_monitor = [CachedImages(args.cache, role, plan)
        for role in ("train", "train_monitor500", "dev_monitor500")]
    model = ReferenceImageGenerator(**plan["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
        patience=config["lr_patience"], threshold=config["min_delta"], threshold_mode="abs",
        min_lr=config["min_lr"])
    state = {"epoch": 0, "cursor": 0, "order": None, "phase": "train", "steps": 0,
        "elapsed_seconds": 0., "best": float("inf"), "significant_best": float("inf"), "stale": 0}
    if args.resume:
        saved = torch.load(args.run / "latest.pt", map_location="cpu", weights_only=False)
        if saved["plan"] != plan:
            raise ValueError("Plan changed: use a registered new run, do not overwrite old state")
        if saved["status"] == "early_stopped":
            raise ValueError("Already early-stopped; extending this run requires a registered migration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        restore_rng(saved["rng"])
        recorded = json.loads((args.run / "identity.json").read_text(encoding="utf-8"))
        for relative, sha in recorded["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != sha:
                raise ValueError("Source changed: register an engine/task migration before resume")
    else:
        args.run.mkdir(parents=True, exist_ok=False)
        atomic_json(args.run / "plan.json", plan)
        identity = {"source_sha256": snapshot(args.run),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_snapshot_includes_uncommitted_changes": True, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "test_read": False, "cache": str(args.cache.resolve())}
        atomic_json(args.run / "identity.json", identity)
    prior_elapsed = state["elapsed_seconds"]
    started = time.perf_counter()

    def save(status, *, best=False):
        if device.type == "cuda":
            torch.cuda.synchronize()
        state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        record = {"plan": plan, "model_config": model.config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "state": state, "rng": rng_state(), "status": status, "test_read": False}
        path = args.run / ("best.pt" if best else "latest.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save(record, temporary)
        temporary.replace(path)
        atomic_json(args.run / "state.json", {**state, "order": "saved_in_latest.pt", "status": status,
            "lr": optimizer.param_groups[0]["lr"], "test_read": False})

    def stop_reason():
        if (args.run / "STOP").exists():
            return "user_stopped"
        if prior_elapsed + time.perf_counter() - started >= config["max_hours"] * 3600:
            return "time_limit"
        return None

    def gallery(name, label):
        if args.gallery is None or stop_reason():
            return
        from .render import publish
        saved_rng = rng_state()
        try:
            publish(model, dev_monitor, args.gallery / name, device=device,
                provenance={"label": label, "run": str(args.run.resolve()),
                    "epoch": state["epoch"], "steps": state["steps"],
                    "best_selection": "dev_monitor500 fixed RGB MSE"}, batch_size=config["batch"])
        finally:
            restore_rng(saved_rng)

    save("running")
    try:
        while state["epoch"] < config["max_epochs"]:
            if reason := stop_reason():
                save(reason)
                return
            epoch_start = time.perf_counter()
            if state["phase"] == "train":
                if state["order"] is None:
                    state["order"] = torch.randperm(len(train)).tolist()
                model.train()
                for batch in loader(train, state["order"][state["cursor"]:], config):
                    if reason := stop_reason():
                        save(reason)
                        return
                    begin = time.perf_counter()
                    batch = move(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = image_loss(model(batch), batch["target"]).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite image training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
                    optimizer.step()
                    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                        raise FloatingPointError("Nonfinite model parameters")
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    state["cursor"] += len(batch["target"])
                    state["steps"] += 1
                    seconds = time.perf_counter() - begin
                    atomic_json(args.run / "heartbeat.json", {"epoch": state["epoch"], "steps": state["steps"],
                        "cursor": state["cursor"], "loss": float(loss.detach()), "gradient_norm": float(norm),
                        "update_seconds": seconds, "update_samples_per_second": len(batch["target"]) / seconds,
                        "test_read": False})
                    if state["steps"] % config["save_steps"] == 0:
                        save("running")
                        if state["steps"] == config["save_steps"]:
                            gallery(f"step_{state['steps']:06d}_preview", f"训练早期预览：第 {state['steps']} 次更新，尚未完成首轮监测")
                state["phase"] = "monitor"
                save("running")
            if reason := stop_reason():
                save(reason)
                return
            train_loss = monitor(model, train_monitor, config, device, stop_reason)
            dev_loss = monitor(model, dev_monitor, config, device, stop_reason)
            scheduler.step(dev_loss)
            improved = dev_loss < state["best"]
            state["best"] = min(state["best"], dev_loss)
            if dev_loss < state["significant_best"] - config["min_delta"]:
                state["significant_best"], state["stale"] = dev_loss, 0
            else:
                state["stale"] += 1
            state["epoch"] += 1
            state["cursor"], state["order"], state["phase"] = 0, None, "train"
            row = {"epoch": state["epoch"], "steps": state["steps"], "train_mse": train_loss,
                "dev_mse": dev_loss, "best": improved, "epoch_segment_seconds": time.perf_counter() - epoch_start,
                "lr_next": optimizer.param_groups[0]["lr"], "test_read": False}
            with (args.run / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if improved:
                save("best", best=True)
                gallery(f"epoch_{state['epoch']:03d}_best", f"第 {state['epoch']} 轮 Loss best，训练仍以登记停止条件结束")
            print(json.dumps(row), flush=True)
            save("running")
            if state["stale"] >= config["early_stop_patience"]:
                save("early_stopped")
                return
        save("epoch_limit")
    except TrainingStopped as stopped:
        save(str(stopped))
        return
    except KeyboardInterrupt:
        # Only a completed update has an advanced cursor. No automatic restart.
        # Use STOP for batch-boundary stopping; an interrupt inside an update
        # leaves the previous atomic latest intact rather than saving partial Adam.
        atomic_json(args.run / "interruption.json", {"status": "interrupted", "restore": "previous_atomic_latest"})
        raise
    except Exception as error:
        # Preserve last known-good state; never overwrite it with a failed update.
        atomic_json(args.run / "error.json", {"type": type(error).__name__, "message": str(error),
            "epoch": state["epoch"], "cursor": state["cursor"], "steps": state["steps"]})
        raise


if __name__ == "__main__":
    main()
