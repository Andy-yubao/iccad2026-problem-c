"""E1-only features, train subset, and supervised targets.

Inference features accept the masked seven-item solver tuple, never labels.
Only load_train_member reads reference geometry, and only frozen train shards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from datasets.cases import allowed_target_positions
from datasets.paths import ROOT, dataset_paths

MEMBERS_PATH = Path(__file__).with_name("train_members.json")
ROWS_PER_SHARD = 112
SHARD_COUNT = 178


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def member_manifest(path=MEMBERS_PATH, create=False):
    """Metadata-only deterministic selection; never inspect held-out tensors."""
    path = Path(path)
    split_bytes = (ROOT / "protocol/split_v1.json").read_bytes()
    split = json.loads(split_bytes)
    train = split["shards"]["train"]
    if set(train) & (set(split["shards"]["dev"]) | set(split["shards"]["local_test"])):
        raise ValueError("Frozen split overlap")
    selected = sorted(train, key=lambda s: (hashlib.sha256(("E1_train_v1:" + s).encode()).hexdigest(), s))[:SHARD_COUNT]
    expected = {
        "version": 1,
        "selection_rule": "first 178 frozen train shards ranked by SHA256('E1_train_v1:' + relative_shard); all rows 0..111",
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "sample_count": SHARD_COUNT * ROWS_PER_SHARD,
        "selected_shards": [{"shard": s, "sample_indices": list(range(ROWS_PER_SHARD))} for s in selected],
    }
    expected["members_sha256"] = digest(expected)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("Training member manifest changed; explicit protocol review required")
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        raise FileNotFoundError(f"Run --prepare-members-only first: {path}")
    return expected


class TrainMembers:
    """One-shard cache and deterministic epoch order, bounded to frozen train."""
    def __init__(self, manifest):
        if manifest != member_manifest():
            raise ValueError("Only the frozen shared train member manifest is accepted")
        self.manifest = manifest
        self.root = dataset_paths()["training_root"]
        self.cached_path = None
        self.cached_raw = None
        self.members = [(entry["shard"], i) for entry in manifest["selected_shards"] for i in entry["sample_indices"]]

    def epoch_order(self, epoch, seed):
        generator = torch.Generator().manual_seed(seed + epoch * 1000003)
        shard_order = torch.randperm(len(self.manifest["selected_shards"]), generator=generator).tolist()
        return [shard * ROWS_PER_SHARD + row for shard in shard_order
                for row in torch.randperm(ROWS_PER_SHARD, generator=generator).tolist()]

    def __getitem__(self, index):
        shard, row = self.members[index]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Train shard escapes root")
        if path != self.cached_path:
            self.cached_raw = torch.load(path, map_location="cpu", weights_only=False)
            if len(self.cached_raw[0]) != ROWS_PER_SHARD:
                raise ValueError("Unexpected row count; manifest must be reviewed")
            self.cached_path = path
        raw = self.cached_raw
        features = raw[0][row].detach().cpu().float()
        areas, constraints = features[:, 0], features[:, 1:]
        reference = raw[5][row][:, [2, 3, 0, 1]].detach().cpu().float()
        n = len(areas)
        if (areas <= 0).any() or reference.shape != (n, 4):
            raise ValueError("Expected positive unpadded training sample")
        specs = allowed_target_positions(reference.tolist(), constraints)
        inputs = (n, areas, raw[1][row].detach().cpu().float(), raw[2][row].detach().cpu().float(),
                  raw[3][row].detach().cpu().float(), constraints, specs)
        return inputs, reference


def graph_features(inputs, device="cpu"):
    n, areas, b2b, p2b, pins, constraints, specs = inputs
    areas, b2b, p2b, pins, c, s = [torch.as_tensor(v, device=device, dtype=torch.float32)
                                   for v in (areas, b2b, p2b, pins, constraints, specs)]
    n = int(n)
    if areas.shape != (n,) or c.shape[0] != n or s.shape != (n, 4) or (areas <= 0).any():
        raise ValueError("Invalid solver tuple")
    if not all(torch.isfinite(v).all() for v in (areas, b2b, p2b, pins, c, s)):
        raise ValueError("Nonfinite solver input")
    # Official rows are unpadded in the observed member. Accept only explicit
    # all-minus-one edge padding; reject negative, fractional or out-of-range IDs.
    cleaned = []
    for edges, first_limit in ((b2b, n), (p2b, len(pins))):
        if edges.ndim != 2 or edges.shape[1] != 3:
            raise ValueError("Expected edge triples")
        edges = edges[~(edges == -1).all(1)]
        ids = edges[:, :2]
        if ((ids < 0).any() or (ids != ids.long()).any() or (ids[:, 0] >= first_limit).any()
                or (ids[:, 1] >= n).any() or (edges[:, 2] < 0).any()):
            raise ValueError("Invalid edge endpoint or weight")
        cleaned.append(edges)
    b2b, p2b = cleaned
    fixed, pre = c[:, 0] != 0, c[:, 1] != 0
    locked = fixed | pre
    if (s[locked, 2:] <= 0).any() or (s[pre, :2] < 0).any():
        raise ValueError("Missing hard specifications")
    # Require masking, so accidental free reference geometry cannot leak in.
    if (s[~locked] != -1).any() or (s[fixed & ~pre, :2] != -1).any():
        raise ValueError("Unmasked ordinary block geometry is forbidden")
    effective = torch.where(locked, s[:, 2:].prod(1), areas)
    scale = effective.sum().sqrt().clamp_min(1e-6)
    known_wh = torch.where(locked[:, None], s[:, 2:] / scale, torch.zeros_like(s[:, 2:]))
    known_center = torch.where(pre[:, None], (s[:, :2] + s[:, 2:] / 2) / scale, torch.zeros_like(s[:, :2]))
    adjacency = torch.zeros((n, n), device=device)
    if b2b.numel():
        i, j = b2b[:, 0].long(), b2b[:, 1].long()
        adjacency.index_put_((i, j), b2b[:, 2].clamp_min(0), accumulate=True)
        adjacency.index_put_((j, i), b2b[:, 2].clamp_min(0), accumulate=True)
    degree = adjacency.sum(1)
    relation = []
    counts = []
    for col in (2, 3):
        same = (c[:, col, None] == c[None, :, col]) & (c[:, col, None] > 0)
        same.fill_diagonal_(False)
        relation.append(same.float() / same.sum(1, keepdim=True).clamp_min(1))
        counts.append(same.sum(1, keepdim=True).float() / max(n, 1))
    pin_weight = torch.zeros(n, device=device)
    pin_center = torch.zeros((n, 2), device=device)
    if p2b.numel():
        pi, bi = p2b[:, 0].long(), p2b[:, 1].long()
        weight = p2b[:, 2].clamp_min(0)
        pin_weight.index_add_(0, bi, weight)
        pin_center.index_add_(0, bi, pins[pi, :2] / scale * weight[:, None])
        pin_center /= pin_weight[:, None].clamp_min(1e-6)
    boundary = ((c[:, 4].long()[:, None] & torch.tensor([1, 2, 4, 8], device=device)) != 0).float()
    features = torch.cat((torch.log(effective / effective.mean()).unsqueeze(1), known_wh, known_center,
                          fixed[:, None].float(), pre[:, None].float(), boundary, *counts,
                          torch.log1p(degree)[:, None], torch.log1p(pin_weight)[:, None], pin_center), dim=1)
    adj = adjacency / degree[:, None].clamp_min(1e-6)
    mask = torch.stack((~pre, ~pre, ~locked), dim=1).float()
    return dict(features=features, adjacency=adj, relations=relation, areas=areas, specs=s,
                scale=scale, fixed=fixed, preplaced=pre, locked=locked, mask=mask)


def encode_target(reference, graph):
    r = torch.as_tensor(reference, device=graph["features"].device)
    return torch.cat(((r[:, :2] + r[:, 2:] / 2) / graph["scale"],
                      torch.log(r[:, 2] / r[:, 3])[:, None]), dim=1)


def decode_layout(state, graph):
    ratio = state[:, 2].clamp(-5, 5)
    wh = torch.stack(((graph["areas"] * ratio.exp()).sqrt(),
                      (graph["areas"] / ratio.exp()).sqrt()), dim=1)
    wh = torch.where(graph["locked"][:, None], graph["specs"][:, 2:], wh)
    xy = (state[:, :2] * graph["scale"] - wh / 2).clamp_min(0)
    out = torch.cat((xy, wh), dim=1)
    return torch.where(graph["preplaced"][:, None], graph["specs"], out)
