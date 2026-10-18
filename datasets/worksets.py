"""Frozen, worker-balanced data membership and raw train/dev access.

Manifest construction uses only split metadata. In particular it never opens
local_test tensors. Features, losses, augmentation and sampling belong to the
consuming experiment, not this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from datasets.paths import ROOT, dataset_paths

WORKSET_PATH = ROOT / "protocol/workset_v2.json"
ROWS_PER_SHARD = 112
SEED = 20261016
PRIMARY = ("train", "dev", "local_test")
PARENTS = {"train": "train", "dev": "dev", "local_test": "local_test",
           "train_monitor": "train", "dev_monitor": "dev", "dev_quality": "dev"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _rank(tag, value):
    return hashlib.sha256(f"workset_v2:{SEED}:{tag}:{value}".encode("utf-8")).hexdigest(), value


def _worker(shard):
    path = Path(shard)
    if len(path.parts) != 2 or not path.parts[0].startswith("worker_"):
        raise ValueError(f"Unexpected frozen shard path: {shard}")
    worker = int(path.parts[0].removeprefix("worker_"))
    if not 0 <= worker < 100 or path.parts[0] != f"worker_{worker}":
        raise ValueError(f"Unexpected worker: {shard}")
    return worker


def _entries(pairs):
    grouped = {}
    for shard, row in pairs:
        grouped.setdefault(shard, []).append(row)
    return [{"shard": shard, "sample_indices": sorted(rows)}
            for shard, rows in sorted(grouped.items(), key=lambda item: (_worker(item[0]), item[0]))]


def _expected_manifest():
    split_bytes = (ROOT / "protocol/split_v1.json").read_bytes()
    split = json.loads(split_bytes)
    pools = {name: set(split["shards"][name]) for name in PRIMARY}
    if any(len(pools[name]) != len(split["shards"][name]) for name in PRIMARY):
        raise ValueError("Duplicate frozen shards")
    if any(pools[a] & pools[b] for a, b in (("train", "dev"), ("train", "local_test"), ("dev", "local_test"))):
        raise ValueError("Frozen shard split overlaps")
    smoke = {(item["shard"], int(item["sample_index"])) for item in split["dev_smoke_panel"]}
    selected = {name: [] for name in PARENTS}
    shard_counts = {"train": 2, "dev": 5, "local_test": 5}
    row_counts = {"train": 100, "dev": 5, "local_test": 5}
    for name in PRIMARY:
        by_worker = {worker: [] for worker in range(100)}
        for shard in pools[name]:
            by_worker[_worker(shard)].append(shard)
        for worker in range(100):
            candidates = by_worker[worker]
            if len(candidates) != (72 if name == "train" else 9):
                raise ValueError(f"Unexpected shard coverage: {name}, worker_{worker}")
            shards = sorted(candidates, key=lambda shard: _rank(f"{name}:shard", shard))[:shard_counts[name]]
            worker_pairs, monitor_pairs = [], []
            for shard in shards:
                rows = [row for row in range(ROWS_PER_SHARD) if name != "dev" or (shard, row) not in smoke]
                rows = sorted(rows, key=lambda row: _rank(f"{name}:row", f"{shard}:{row}"))[:row_counts[name]]
                pairs = [(shard, row) for row in rows]
                worker_pairs.extend(pairs)
                if name == "dev":
                    monitor_pairs.append(min(pairs, key=lambda pair: _rank("dev_monitor", f"{pair[0]}:{pair[1]}")))
            selected[name].extend(worker_pairs)
            if name == "train":
                selected["train_monitor"].extend(sorted(worker_pairs, key=lambda pair: _rank("train_monitor", f"{pair[0]}:{pair[1]}"))[:5])
            elif name == "dev":
                selected["dev_monitor"].extend(monitor_pairs)
                selected["dev_quality"].append(min(monitor_pairs, key=lambda pair: _rank("dev_quality", f"{pair[0]}:{pair[1]}")))
    expected = {
        "version": 2,
        "seed": SEED,
        "split_file": "protocol/split_v1.json",
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "rows_per_shard": ROWS_PER_SHARD,
        "selection_rule": {
            "ranking": "Ascending (SHA256(UTF8('workset_v2:20261016:' + tag + ':' + value)), value).",
            "shards": "Within workers 0..99 rank frozen split shards using tag '<split>:shard', value relative shard. Select train=2, dev=5, local_test=5 per worker.",
            "rows": "Within each selected shard rank rows 0..111 using tag '<split>:row', value '<shard>:<row>'. Before dev row ranking exclude exact frozen dev_smoke_panel members. Select train=100, dev=5, local_test=5 per shard.",
            "train_monitor": "Within each worker rank its 200 selected train pairs using tag 'train_monitor', value '<shard>:<row>'; select first 5.",
            "dev_monitor": "Within each selected dev shard rank its 5 selected pairs using tag 'dev_monitor', value '<shard>:<row>'; select first 1.",
            "dev_quality": "Within each worker rank its 5 dev_monitor pairs using tag 'dev_quality', value '<shard>:<row>'; select first 1.",
            "serialization": "Entries ordered by numeric worker then shard path; row indices ascending. Canonical checksum is SHA256 of UTF8 JSON with sorted keys, compact separators and ensure_ascii=False, excluding manifest_sha256.",
        },
        "sealed_sets": ["local_test"],
        "excluded_dev_smoke_members": [{"shard": shard, "sample_index": row} for shard, row in sorted(smoke)],
        "counts": {name: len(pairs) for name, pairs in selected.items()},
        "members": {name: _entries(pairs) for name, pairs in selected.items()},
    }
    expected["manifest_sha256"] = _digest(expected)
    return expected


def workset_manifest(path=WORKSET_PATH, create=False):
    """Create/read metadata only, refusing any existing noncanonical manifest."""
    path = Path(path)
    expected = _expected_manifest()
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Frozen workset differs from canonical membership: {path}; explicit protocol revision required")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation cannot silently overwrite another writer's result.
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    else:
        raise FileNotFoundError(f"Prepare workset metadata first: {path}")
    return expected


def member_pairs(manifest, name):
    """Explicit sample membership; local_test metadata is allowed, data is sealed."""
    if name not in PARENTS:
        raise ValueError(f"Unknown workset: {name}")
    return [(entry["shard"], int(row)) for entry in manifest["members"][name]
            for row in entry["sample_indices"]]


@dataclass
class RawLayoutCase:
    case_id: str
    source: dict
    solver_inputs: tuple
    reference_positions: list
    stored_metrics: object

    @property
    def block_count(self):
        return self.solver_inputs[0]


class LayoutReader:
    """One-shard CPU cache; local_test requires an explicit final-evaluation gate.

    __getitem__ returns (masked_solver_inputs, reference_tensor), matching the
    original E1 TrainMembers interface. Training keeps allow_local_test=False.
    A user-authorized frozen evaluation can explicitly enable local_test.
    case() returns raw stored metrics too;
    this module does not score layouts or compute any model-specific feature.
    """
    def __init__(self, manifest, name, *, allow_local_test=False):
        if name not in PARENTS or (PARENTS[name] == "local_test" and not allow_local_test):
            raise ValueError("Only train/dev worksets may be read; local_test is sealed")
        if manifest != _expected_manifest():
            raise ValueError("Only the canonical frozen workset is accepted")
        self.manifest, self.name = manifest, name
        self.members = tuple(member_pairs(manifest, name))
        self._allowed_members = frozenset(self.members)
        self.root = dataset_paths()["training_root"]
        self.cached_path = self.cached_raw = None

    def __len__(self):
        return len(self.members)

    def _read(self, index):
        import torch
        from datasets.cases import allowed_target_positions
        shard, row = self.members[index]
        if (shard, row) not in self._allowed_members:
            raise ValueError("Requested layout is outside the selected train/dev workset")
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Workset shard escapes training root")
        if path != self.cached_path:
            self.cached_raw = torch.load(path, map_location="cpu", weights_only=False)
            if len(self.cached_raw) < 7 or len(self.cached_raw[0]) != ROWS_PER_SHARD:
                raise ValueError("Unexpected raw schema/row count; frozen workset requires review")
            self.cached_path = path
        raw = self.cached_raw
        features = raw[0][row].detach().cpu().float().clone()
        if features.ndim != 2 or features.shape[1] < 6:
            raise ValueError("Expected block area plus five constraint columns")
        areas, constraints = features[:, 0], features[:, 1:]
        reference = raw[5][row][:, [2, 3, 0, 1]].detach().cpu().float().clone()
        n = len(areas)
        if n == 0 or (areas <= 0).any() or reference.shape != (n, 4) or (reference[:, 2:] <= 0).any():
            raise ValueError("Expected positive, unpadded raw layout")
        specs = allowed_target_positions(reference.tolist(), constraints)
        edges_and_pins = [raw[column][row].detach().cpu().float().clone() for column in (1, 2, 3)]
        if not all(torch.isfinite(value).all() for value in (features, reference, specs, *edges_and_pins)):
            raise ValueError("Nonfinite raw layout")
        fixed, preplaced = constraints[:, 0] != 0, constraints[:, 1] != 0
        if (specs[~(fixed | preplaced)] != -1).any() or (specs[fixed & ~preplaced, :2] != -1).any():
            raise ValueError("Unmasked reference geometry")
        inputs = (n, areas, *edges_and_pins, constraints, specs)
        metrics = raw[6][row].detach().cpu().clone()
        source = {"split": PARENTS[self.name], "workset": self.name, "shard": shard,
                  "sample_index": row, "paths": [path.relative_to(ROOT).as_posix()],
                  "workset_sha256": self.manifest["manifest_sha256"]}
        return inputs, reference, metrics, source

    def __getitem__(self, index):
        inputs, reference, _, _ = self._read(index)
        return inputs, reference

    def case(self, index):
        inputs, reference, metrics, source = self._read(index)
        return RawLayoutCase(f"{source['split']}:{source['shard']}:{source['sample_index']}",
                             source, inputs, reference.tolist(), metrics)
