"""Check exact outputs and compare CPU time against the accepted postprocessor."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import permutations
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

# benchmark sets the native thread limits before importing NumPy and PyTorch.
from .benchmark import FROZEN, PLAN, RandomReference, repair_inputs, sha

import numpy as np

from experiments.E165_joint_l0.evaluation.solver import optimize as e165

from .optimizer import optimize as accelerated
from .optimizer import optimize_reference as accepted


RUNTIME_SAMPLE_SEED = 20261125
RUNTIME_SAMPLE_SIZE = 120


def timed(optimize, draft, inputs):
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    layout, info = optimize(draft, inputs)
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
    return layout, info, cpu_ms, wall_ms


def run_case(index):
    ref = dataset.records[index]
    if sha(dataset.folder / ref["payload"]) != checksums[ref["id"]]:
        raise ValueError(f"Dev payload changed at case {index}")
    inputs = repair_inputs(dataset[index]["solver_inputs"])
    with np.load(args.original / f"{index:04d}.npz", allow_pickle=False) as original:
        draft, e165_expected = original["draft"], original["final"]
    with np.load(args.current / f"{index:04d}.npz", allow_pickle=False) as current:
        if not np.array_equal(draft, current["draft"]):
            raise ValueError(f"Cached draft differs at case {index}")
        expected = current["final"]
        metrics = json.loads(current["metrics"].item())

    methods = (
        (accepted, accelerated, e165) if args.serial_sample else (accepted, accelerated)
    )
    orders = tuple(permutations(range(len(methods))))
    order = orders[index % len(orders)]
    measured = {which: timed(methods[which], draft, inputs) for which in order}
    before, before_info, before_cpu, before_wall = measured[0]
    after, after_info, after_cpu, after_wall = measured[1]
    if not np.array_equal(before.view(np.uint64), expected.view(np.uint64)):
        raise AssertionError(
            f"Accepted version differs from its frozen output at case {index}"
        )
    if not np.array_equal(after.view(np.uint64), before.view(np.uint64)):
        max_delta = np.max(np.abs(after - before))
        raise AssertionError(
            f"Accelerated layout differs at case {index}, max delta {max_delta}"
        )
    if before_info["baseline"]["work"] != after_info["baseline"]["work"]:
        raise AssertionError(f"Accelerated work count differs at case {index}")
    stages = (
        "soft_moves",
        "legal_seed_attempted",
        "legal_seed_accepted",
        "anchor_seed_attempted",
        "anchor_seed_accepted",
    )
    if any(before_info["refinement"][k] != after_info["refinement"][k] for k in stages):
        raise AssertionError(f"Accelerated soft repair differs at case {index}")

    row = dict(
        index=index,
        blocks=inputs[0],
        q=metrics["q"],
        grouping=metrics["group"],
        boundary=metrics["boundary"],
        exact=True,
        current_cpu_ms=before_cpu,
        accelerated_cpu_ms=after_cpu,
        current_wall_ms=before_wall,
        accelerated_wall_ms=after_wall,
    )
    if args.serial_sample:
        e165_layout, _, e165_cpu, e165_wall = measured[2]
        if not np.array_equal(
            e165_layout.view(np.uint64), e165_expected.view(np.uint64)
        ):
            raise AssertionError(f"E165 differs from its frozen output at case {index}")
        row.update(e165_cpu_ms=e165_cpu, e165_wall_ms=e165_wall)
    return row


def latency(rows, clock):
    def distribution(variant):
        times = [row[f"{variant}_{clock}_ms"] for row in rows]
        return dict(
            mean=float(np.mean(times)),
            median=float(np.median(times)),
            p95=float(np.percentile(times, 95)),
            p99=float(np.percentile(times, 99)),
            max=float(np.max(times)),
        )

    current, faster = distribution("current"), distribution("accelerated")
    summary = dict(
        current=current,
        accelerated=faster,
        reduction_pct={key: 100 * (1 - faster[key] / current[key]) for key in current},
    )
    if f"e165_{clock}_ms" in rows[0]:
        summary["e165_original"] = distribution("e165")
    return summary


def emit(rows):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: row["index"])
    fields = [
        name for name in rows[0] if args.serial_sample or not name.endswith("_wall_ms")
    ]
    label = "serial120" if args.serial_sample else "dev2000"
    with (out / f"ICCAD_postprocess_speed_{label}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        count=len(rows),
        identical=sum(row["exact"] for row in rows),
        workers=args.workers,
        cpu_ms=latency(rows, "cpu"),
        checkpoint_sha256=FROZEN["checkpoint_sha256"],
        payload_checksums_sha256=PLAN["payload_checksums_sha256"],
    )
    if args.serial_sample:
        result["wall_ms"] = latency(rows, "wall")
        result["sample_seed"] = RUNTIME_SAMPLE_SEED
    (out / f"speed_{label}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def collect(results, total):
    rows = []
    for row in results:
        rows.append(row)
        count = len(rows)
        if count == 1 or count % 100 == 0 or count == total:
            print(f"{count}/{total} exact layouts", flush=True)
    return rows


def main():
    global args, dataset, checksums
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serial-sample", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is None:
        args.workers = 1 if args.serial_sample or os.name == "nt" else 7
    if args.workers < 1 or (
        args.workers > 1 and (args.serial_sample or os.name == "nt")
    ):
        parser.error("Use one worker for serial timing or on Windows")
    dataset = RandomReference("dev")
    if [row["member"] for row in dataset.records] != PLAN["members"]:
        raise ValueError("Frozen dev2000 members changed")
    ledger = dataset.folder / "checksums.jsonl"
    if sha(ledger) != PLAN["payload_checksums_sha256"]:
        raise ValueError("Frozen dataset checksum ledger changed")
    checksums = {
        record["id"]: record["payload_sha256"]
        for line in ledger.read_text().splitlines()
        for record in json.loads(line)["records"]
        if record["role"] == "dev"
    }
    run_case(
        575
    )  # Warm the native kernels without including their first call in latency.
    indices = list(range(len(dataset)))
    if args.serial_sample:
        indices = random.Random(RUNTIME_SAMPLE_SEED).sample(
            indices, RUNTIME_SAMPLE_SIZE
        )
    if args.workers == 1:
        emit(collect(map(run_case, indices), len(indices)))
    else:
        with ProcessPoolExecutor(
            args.workers, mp_context=mp.get_context("fork")
        ) as pool:
            emit(collect(pool.map(run_case, indices), len(indices)))


if __name__ == "__main__":
    main()
