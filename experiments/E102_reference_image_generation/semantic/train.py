"""One formal semantic-v2.1 run, fixed two-view objective and frozen monitors."""
import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.paths import ROOT
from ..file_io import atomic_json, advisory_json, advisory_append, replace_file
from .data import CachedSemanticImages
from .generation import corrupt
from .loss import semantic_loss
from .model import SemanticGenerator, MASK
from .render import publish, image_scores
from .runtime import TrainingStopped, get_rng, set_rng, snapshot
from .precision import forward_image


def loader(data, indices, config):
    return DataLoader(Subset(data, indices), batch_size=config["batch"], shuffle=False,
        num_workers=config["workers"], pin_memory=True, generator=torch.Generator().manual_seed(config["seed"]))


def tensors(batch, device):
    return ({k: batch[k].to(device, non_blocking=True) for k in ("features", "relations", "valid")},
            batch["labels"].to(device, non_blocking=True))


def monitor_noise(target, indices, dataset):
    states, fractions, hidden = [], [], []
    for i, member_index in enumerate(indices):
        member = dataset.records[int(member_index)]["member"]
        seed = int(hashlib.sha256(f"E102.v21.monitor:{member[0]}:{member[1]}".encode()).hexdigest()[:15], 16)
        generator = torch.Generator(device=target.device).manual_seed(seed)
        noisy, fraction, mask = corrupt(target[i:i + 1], generator)
        states.append(noisy)
        fractions.append(fraction)
        hidden.append(mask)
    return torch.cat(states), torch.cat(fractions), torch.cat(hidden)


@torch.no_grad()
def monitor(model, dataset, config, device, should_stop):
    model.eval()
    totals = dict(loss=0., all_masked_loss=0., random_masked_loss=0., ce=0., dice=0., boundary=0.,
                  module_mean_IoU=0., foreground_pixel_accuracy=0.)
    count = 0
    for batch in loader(dataset, list(range(len(dataset))), config):
        if reason := should_stop():
            raise TrainingStopped(reason)
        public, target = tensors(batch, device)
        full = torch.full_like(target, MASK)
        output = forward_image(model, full, torch.ones(len(target), device=device), public, config["precision"])
        all_loss, all_parts = semantic_loss(output, target, torch.ones_like(target, dtype=torch.bool))
        scores = image_scores(output["logits"].argmax(1), target)
        del output
        noisy, fraction, hidden = monitor_noise(target, batch["index"], dataset)
        output = forward_image(model, noisy, fraction, public, config["precision"])
        random_loss, random_parts = semantic_loss(output, target, hidden)
        loss = .5 * (all_loss + random_loss)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite semantic monitoring loss")
        values = {"loss": loss, "all_masked_loss": all_loss, "random_masked_loss": random_loss,
            **{k: .5 * (all_parts[k] + random_parts[k]) for k in all_parts}, **scores}
        sums = torch.stack([values[k].sum() for k in totals]).cpu().tolist()
        for key, value in zip(totals, sums):
            totals[key] += value
        count += len(target)
    return {k: v / count for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "run", "cache", "gallery"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migrate-from", type=Path)
    parser.add_argument("--migration", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan["formal_training_authorized"]:
        raise PermissionError("Formal training requires explicit run authorization")
    config = plan["training"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    datasets = {role: CachedSemanticImages(args.cache, role) for role in ("train", "train_monitor500", "dev_monitor500")}
    train, train_set, dev_set = [datasets[k] for k in ("train", "train_monitor500", "dev_monitor500")]
    model = SemanticGenerator(**plan["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
        patience=config["lr_patience"], threshold=config["min_delta"], threshold_mode="abs", min_lr=config["min_lr"])
    corruption_rng = torch.Generator(device=device).manual_seed(config["seed"])
    state = {"epoch": 0, "steps": 0, "cursor": 0, "order": None, "phase": "initial_monitor",
        "monitor_id": 0, "best": float("inf"), "significant_best": float("inf"), "stale": 0,
        "elapsed_seconds": 0., "seen_count": 0}
    state["epoch_started_elapsed"] = 0.
    parent_latest = parent_best = None
    if args.migrate_from:
        if args.resume or not args.migration:
            raise ValueError("Migration uses a new run and an explicit migration record")
        migration = json.loads(args.migration.read_text(encoding="utf-8"))
        for name in ("latest", "best"):
            if hashlib.sha256((args.migrate_from / f"{name}.pt").read_bytes()).hexdigest() != migration[f"parent_{name}_sha256"]:
                raise ValueError("Parent checkpoint differs from registered migration")
        parent_latest = torch.load(args.migrate_from / "latest.pt", map_location="cpu", weights_only=False)
        parent_best = torch.load(args.migrate_from / "best.pt", map_location="cpu", weights_only=False)
        for key in ("task", "architecture", "loss", "corruption", "area_transform"):
            if parent_latest["plan"][key] != plan[key]:
                raise ValueError("Engineering migration changed scientific target: " + key)
        parent_identity = json.loads((args.migrate_from / "identity.json").read_text(encoding="utf-8"))
        for name in migration["unchanged_math_sources"]:
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != parent_identity["source_sha256"][name]:
                raise ValueError("Model/Loss/data source changed: " + name)
        model.load_state_dict(parent_latest["model"])
        optimizer.load_state_dict(parent_latest["optimizer"])
        scheduler.load_state_dict(parent_latest["scheduler"])
        state = copy.deepcopy(parent_latest["state"])
        set_rng(parent_latest["rng"])
        corruption_rng.set_state(parent_latest["corruption_rng"])
        state["migration_resume_phase"] = state["phase"]
        state["phase"] = "migration_baseline"
        state["next_monitor_seen"] = migration["next_monitor_seen"]
        state["migration_start_steps"] = state["steps"]
        state["parent_best_loss_fp32"] = state["best"]
        state["engine"] = "BF16_batch4_FP32_loss_Adam.v1"
    elif args.resume:
        saved = torch.load(args.run / "latest.pt", map_location="cpu", weights_only=False)
        if saved["plan"] != plan or saved["status"] in ("early_stopped", "epoch_limit", "time_limit"):
            raise ValueError("Resume must retain the plan and cannot extend a completed budget")
        identity = json.loads((args.run / "identity.json").read_text(encoding="utf-8"))
        for name, sha in identity["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Source changed: register migration before resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        set_rng(saved["rng"])
        corruption_rng.set_state(saved["corruption_rng"])
    if not args.resume:
        args.run.mkdir(parents=True, exist_ok=False)
        atomic_json(args.run / "plan.json", plan)
        atomic_json(args.run / "identity.json", {"source_sha256": snapshot(args.run),
            "cache_metadata_sha256": {k: hashlib.sha256((args.cache / k / "metadata.json").read_bytes()).hexdigest() for k in datasets},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_snapshot_includes_uncommitted_changes": True, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "initialization": str(args.migrate_from.resolve()) if args.migrate_from else "fresh seed0",
            "migration": str(args.migration.resolve()) if args.migration else None, "test_read": False})
    prior_elapsed, started = state["elapsed_seconds"], time.perf_counter()

    def reason_to_stop():
        if (args.run / "STOP").exists() or (args.plan.parent / "STOP").exists():
            return "user_stopped"
        if prior_elapsed + time.perf_counter() - started >= config["max_hours"] * 3600:
            return "time_limit"
        return None

    def save(status, best=False):
        torch.cuda.synchronize()
        state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - started
        record = {"model": model.state_dict(), "model_config": model.config, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": get_rng(), "corruption_rng": corruption_rng.get_state(),
            "state": state, "plan": plan, "status": status, "test_read": False}
        path = args.run / ("best.pt" if best else "latest.pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save(record, temporary)
        replace_file(temporary, path, timeout=5.)
        advisory_json(args.run / "state.json", {**state, "order": "saved_in_latest.pt", "status": status,
            "lr": optimizer.param_groups[0]["lr"], "test_read": False})

    def images(name, label):
        if reason := reason_to_stop():
            raise TrainingStopped(reason)
        rng = get_rng()
        try:
            publish(model, dev_set, args.gallery, name, device=device, steps=state["steps"],
                    monitor_id=state["monitor_id"], label=label, should_stop=reason_to_stop, precision=config["precision"])
        finally:
            set_rng(rng)

    next_heartbeat = 0.
    save("running")
    try:
        while state["epoch"] < config["max_epochs"]:
            if reason := reason_to_stop():
                raise TrainingStopped(reason)
            if state["phase"] == "migration_baseline":
                if parent_latest is None:
                    parent_path = Path(plan["migration"]["parent_run"])
                    parent_latest = torch.load(parent_path / "latest.pt", map_location="cpu", weights_only=False)
                    parent_best = torch.load(parent_path / "best.pt", map_location="cpu", weights_only=False)
                # Recalibrate only existing frozen checkpoints in the new numeric
                # path; do not advance the scheduler or erase bad-event counters.
                model.load_state_dict(parent_best["model"])
                best_values = monitor(model, dev_set, config, device, reason_to_stop)
                best_record = copy.deepcopy(parent_best)
                best_record["plan"], best_record["status"] = plan, "best_migrated_to_bf16"
                best_record["state"]["best"] = best_values["loss"]
                torch.save(best_record, args.run / "best.pt.tmp")
                replace_file(args.run / "best.pt.tmp", args.run / "best.pt", timeout=5.)
                model.load_state_dict(parent_latest["model"])
                train_values = monitor(model, train_set, config, device, reason_to_stop)
                dev_values = monitor(model, dev_set, config, device, reason_to_stop)
                offset = best_values["loss"] - state["parent_best_loss_fp32"]
                scheduler.best += offset
                state["significant_best"] += offset
                state["best"] = best_values["loss"]
                latest_better = dev_values["loss"] < state["best"]
                if latest_better:
                    state["best"] = dev_values["loss"]
                state["phase"] = state.pop("migration_resume_phase")
                calibration = {"event": "precision_migration_baseline", "steps": state["steps"],
                    "seen_count": state["seen_count"], "train": train_values, "dev": dev_values,
                    "train_loss": train_values["loss"], "dev_loss": dev_values["loss"],
                    "inherited_best_bf16": best_values, "precision": config["precision"],
                    "lr": optimizer.param_groups[0]["lr"], "stale_preserved": state["stale"],
                    "scheduler_bad_events_preserved": scheduler.num_bad_epochs, "test_read": False}
                atomic_json(args.run / "migration_baseline.json", calibration)
                advisory_append(args.run / "progress.jsonl", calibration)
                print(json.dumps(calibration), flush=True)
                if latest_better:
                    save("best", best=True)
                save("running")
                del parent_latest, parent_best
                parent_latest = parent_best = None
                continue
            if state["phase"] in ("initial_monitor", "monitor"):
                initial = state["phase"] == "initial_monitor"
                before = time.perf_counter()
                train_values = monitor(model, train_set, config, device, reason_to_stop)
                dev_values = monitor(model, dev_set, config, device, reason_to_stop)
                scheduler.step(dev_values["loss"])
                improved = dev_values["loss"] < state["best"]
                state["best"] = min(state["best"], dev_values["loss"])
                if dev_values["loss"] < state["significant_best"] - config["min_delta"]:
                    state["significant_best"], state["stale"] = dev_values["loss"], 0
                else:
                    state["stale"] += 1
                if not initial:
                    state["monitor_id"] += 1
                if config.get("monitor_seen_interval"):
                    while state["next_monitor_seen"] <= state["seen_count"]:
                        state["next_monitor_seen"] += config["monitor_seen_interval"]
                row = {"event": "monitor", "monitor_id": state["monitor_id"], "steps": state["steps"],
                    "epoch_fraction": state["epoch"] + state["cursor"] / len(train), "train_loss": train_values["loss"],
                    "dev_loss": dev_values["loss"], "train": train_values, "dev": dev_values,
                    "best": improved, "monitor_seconds": time.perf_counter() - before,
                    "lr_next": optimizer.param_groups[0]["lr"], "test_read": False}
                if state["cursor"] == len(train):
                    row["full_epoch_wall_seconds"] = prior_elapsed + time.perf_counter() - started - state["epoch_started_elapsed"]
                advisory_append(args.run / "progress.jsonl", row)
                print(json.dumps(row), flush=True)
                if state["cursor"] == len(train):
                    state["epoch"] += 1
                    state["cursor"], state["order"] = 0, None
                    state["epoch_started_elapsed"] = prior_elapsed + time.perf_counter() - started
                state["phase"] = "train"
                if improved:
                    save("best", best=True)
                save("running")
                if improved and not initial:
                    images(f"monitor_{state['monitor_id']:03d}_best", f"固定dev500双视图Loss best；第{state['steps']}次更新")
                if state["stale"] >= config["early_stop_patience"]:
                    save("early_stopped")
                    return
                continue
            if state["order"] is None:
                state["order"] = torch.randperm(len(train)).tolist()
            model.train()
            previous_end = time.perf_counter()
            for batch in loader(train, state["order"][state["cursor"]:], config):
                if reason := reason_to_stop():
                    raise TrainingStopped(reason)
                began = time.perf_counter()
                prepare_seconds = began - previous_end
                public, target = tensors(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = forward_image(model, torch.full_like(target, MASK), torch.ones(len(target), device=device), public, config["precision"])
                full_loss, _ = semantic_loss(output, target, torch.ones_like(target, dtype=torch.bool))
                if not torch.isfinite(full_loss).all():
                    raise FloatingPointError("Nonfinite all-MASK loss")
                (.5 * full_loss.mean()).backward()
                full_value = float(full_loss.detach().mean())
                del output, full_loss
                noisy, fraction, hidden = corrupt(target, corruption_rng)
                output = forward_image(model, noisy, fraction, public, config["precision"])
                masked_loss, _ = semantic_loss(output, target, hidden)
                if not torch.isfinite(masked_loss).all():
                    raise FloatingPointError("Nonfinite random-MASK loss")
                (.5 * masked_loss.mean()).backward()
                random_value = float(masked_loss.detach().mean())
                del output, masked_loss
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
                optimizer.step()
                if not torch.stack([p.isfinite().all() for p in model.parameters()]).all():
                    raise FloatingPointError("Nonfinite parameters")
                torch.cuda.synchronize()
                state["steps"] += 1
                state["cursor"] += len(target)
                state["seen_count"] += len(target)
                update_seconds = time.perf_counter() - began
                heartbeat = {"steps": state["steps"], "epoch": state["epoch"], "cursor": state["cursor"],
                    "seen_count": state["seen_count"], "loss": .5 * (full_value + random_value),
                    "all_masked_loss": full_value, "random_masked_loss": random_value, "gradient_norm": float(grad),
                    "update_seconds": update_seconds, "batch_prepare_seconds": prepare_seconds,
                    "update_samples_per_second": len(target) / update_seconds,
                    "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20, "test_read": False}
                if time.monotonic() >= next_heartbeat:
                    published = advisory_json(args.run / "heartbeat.json", heartbeat)
                    next_heartbeat = time.monotonic() + (1. if published else 10.)
                if state["steps"] % 50 == 0:
                    advisory_append(args.run / "updates.jsonl", heartbeat)
                    print(json.dumps({"event": "update", **heartbeat}), flush=True)
                if state["steps"] % config["save_steps"] == 0:
                    save("running")
                if state["steps"] == 100 or state["steps"] == state.get("migration_start_steps", -100) + 100:
                    images(f"step_{state['steps']:06d}_preview", f"累计第{state['steps']}次更新预览；以完整dev500监测作选择")
                due = (state["seen_count"] >= state["next_monitor_seen"] if config.get("monitor_seen_interval")
                       else state["steps"] % config["monitor_steps"] == 0)
                if due or state["cursor"] == len(train):
                    state["phase"] = "monitor"
                    save("running")
                    break
                previous_end = time.perf_counter()
        save("epoch_limit")
    except TrainingStopped as stopped:
        save(str(stopped))
    except KeyboardInterrupt:
        atomic_json(args.run / "interruption.json", {"restore": "previous atomic latest; interrupted update not saved"})
        raise
    except Exception as error:
        atomic_json(args.run / "error.json", {"type": type(error).__name__, "message": str(error),
            "steps": state["steps"], "cursor": state["cursor"]})
        raise


if __name__ == "__main__":
    main()
