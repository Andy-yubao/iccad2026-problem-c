"""User-run bounded E2 training; consumes E1's frozen train-members JSON only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, default=ROOT / "experiments/graph_proposal/train_members.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3600.)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument('--legacy-protocol',action='store_true',help='Historical reproduction only; current default requires workset_v2 and validation controls')
    args = parser.parse_args()
    if not args.legacy_protocol:
        parser.error('This historical trainer must be upgraded to workset_v2 and validation controls before new research training; --legacy-protocol is for explicit reproduction only.')
    if not 0 < args.seconds <= 3600 or args.max_steps < 1:
        parser.error("E2 first-round budget is >0 and <=3600 seconds, positive max steps")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    import numpy as np
    from datasets.paths import dataset_paths
    from datasets.cases import allowed_target_positions
    from experiments.topology_geometry.input_geometry import prepare_problem
    from experiments.topology_geometry.geometry import fixed_shapes
    from experiments.topology_geometry.model import RelationModel, features, ambiguous_targets, ambiguity_loss
    from experiments.topology_geometry.run import source_record
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    if not args.members.is_file():
        raise FileNotFoundError("Prepare the common E1 train member manifest first; see README")
    members_bytes = args.members.read_bytes()
    members_hash = hashlib.sha256(members_bytes).hexdigest()
    manifest = json.loads(members_bytes)
    split = json.loads((ROOT / "protocol/split_v1.json").read_text(encoding="utf-8"))
    entries = manifest["selected_shards"]
    count = sum(len(e["sample_indices"]) for e in entries)
    if count > 20000 or count < 1:
        raise ValueError("Common train subset must contain 1..20000 layouts")
    seen = set()
    for entry in entries:
        if entry["shard"] not in split["shards"]["train"]:
            raise ValueError("Member does not belong to frozen train split")
        for index in entry["sample_indices"]:
            key = entry["shard"], index
            if not isinstance(index, int) or not 0 <= index < 112 or key in seen:
                raise ValueError("Invalid/duplicate train layout member")
            seen.add(key)
    if args.output.exists() and not args.resume:
        raise FileExistsError("Do not overwrite an existing training run")
    # Validate the checkpoint before any mutation of its original run. Existing
    # snapshots are read-only; source_record also refuses mismatching contents.
    current_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(Path(__file__).parent.glob("*.py"))}
    state = None
    if args.resume:
        if args.resume.resolve().parent != args.output.resolve():
            raise ValueError("Resume must reference this output run's checkpoint")
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        if state.get("kind") != "topology_geometry_relations_v1":
            raise ValueError("Not an E2 relation checkpoint")
        if state["members_sha256"] != members_hash or state["seed"] != args.seed:
            raise ValueError("Resume manifest or seed changed")
        if state["source"]["hashes"] != current_hashes:
            raise ValueError("Resume source changed; preserve run and review explicitly")
        if not (args.output / "source.zip").is_file():
            raise FileNotFoundError("Resume requires the original source snapshot")
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    source = source_record(args.output, "source.zip")
    paths = dataset_paths(ROOT)
    model = RelationModel().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    config = {"command": sys.argv, "seed": args.seed, "device": args.device, "threads": 1,
              "members": str(args.members.resolve()), "members_sha256": members_hash,
              "member_count": count, "seconds_budget": args.seconds, "max_steps": args.max_steps,
              "data_paths": {k: str(v) for k, v in paths.items()}, "source": source,
              "torch": torch.__version__, "numpy": np.__version__,
              "target": "marginal probability of every valid label separation direction",
              "label_shapes_in_input": False}
    step, elapsed_before, offset, epoch = 0, 0., 0, 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, elapsed_before, epoch, offset = state["step"], state["elapsed_seconds"], state["epoch"], state["offset"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if args.device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    config_path = args.output / (f"resume_{step}_{time.time_ns()}.json" if args.resume else "config.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    cache_shard, raw = None, None
    last_loss = None
    reason = "resource_stop"
    order = sorted(seen)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    def checkpoint(reason):
        elapsed = elapsed_before + time.perf_counter() - started
        payload = {"kind": "topology_geometry_relations_v1", "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "members_sha256": members_hash,
                   "seed": args.seed, "step": step, "elapsed_seconds": elapsed,
                   "epoch": epoch, "offset": offset, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
                   "source": source, "stop_reason": reason, "last_loss": last_loss}
        temp = args.output / "latest.tmp.pt"
        torch.save(payload, temp)
        temp.replace(args.output / "latest.pt")
        (args.output / "status.json").write_text(json.dumps({k: payload[k] for k in
            ("step", "elapsed_seconds", "epoch", "offset", "stop_reason", "last_loss")}, indent=2), encoding="utf-8")
    try:
        # This initial state and periodic saves contain completed optimizer steps.
        # A runtime exception leaves the most recent complete checkpoint intact.
        checkpoint("resumed" if args.resume else "initialized")
        while (not stop_requested and step < args.max_steps
               and elapsed_before + time.perf_counter() - started < args.seconds - 5):
            if offset >= len(order):
                epoch, offset = epoch + 1, 0
            shard, index = order[offset]
            path = (paths["training_root"] / shard).resolve()
            if not path.is_relative_to(paths["training_root"]):
                raise ValueError("Invalid train shard path")
            if cache_shard != shard:
                raw = torch.load(path, map_location="cpu", weights_only=False)
                cache_shard = shard
            if stop_requested or elapsed_before + time.perf_counter() - started >= args.seconds - 5:
                break
            block = raw[0][index]
            reference = raw[5][index][:, [2, 3, 0, 1]].tolist()
            constraints = block[:, 1:]
            safe = (len(block), block[:, 0], raw[1][index], raw[2][index], raw[3][index], constraints,
                    allowed_target_positions(reference, constraints))
            p = prepare_problem(*safe)
            data = features(p, fixed_shapes(p), args.device)
            valid = ambiguous_targets(reference).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = ambiguity_loss(model(*data), valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            last_loss = float(loss.detach())
            step, offset = step + 1, offset + 1
            if step % 100 == 0:
                checkpoint("running")
                print(json.dumps({"step": step, "loss": last_loss, "elapsed_seconds": elapsed_before + time.perf_counter() - started}), flush=True)
        reason = ("user_interrupt_between_batches" if stop_requested else
                  "max_steps_resource_stop" if step >= args.max_steps else "time_budget_resource_stop")
        checkpoint(reason)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(json.dumps({"output": str(args.output.resolve()), "reason": reason, "step": step}), flush=True)


if __name__ == "__main__":
    main()
