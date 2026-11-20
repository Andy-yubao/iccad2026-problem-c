"""Raw, unmodified v5 members; public conditioning and separate image labels."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.reference_layouts import registry
from datasets.worksets import FrozenMemberReader
from visualization.raster_layouts import module_palette, reference_rgb

MAX_MODULES = 120
FEATURES = 22


def public_features(inputs, *, kappa, margin, include_terminal_extent=False):
    n, areas, b2b, p2b, pins, c, s = inputs
    areas, b2b, p2b, pins, c, s = [torch.as_tensor(x).float() for x in (areas, b2b, p2b, pins, c, s)]
    n = int(n)
    if not 1 <= n <= MAX_MODULES or c.shape != (n, 5) or s.shape != (n, 4):
        raise ValueError("Unexpected public schema")
    locked, pre = c[:, :2].bool().any(1), c[:, 1].bool()
    if (s[~locked] != -1).any() or (s[locked & ~pre, :2] != -1).any():
        raise ValueError("Free reference geometry must not enter conditioning")
    effective = torch.where(locked, s[:, 2:].prod(1), areas)
    scale = float(effective.sum().sqrt())
    # Fixed physical origin; do not fit or translate by free-label extents.
    extent = float((s[pre, :2] + s[pre, 2:]).max()) if pre.any() else 0.
    if include_terminal_extent and pins.numel():
        extent = max(extent, float(pins[:, :2].max()))
    side = max(kappa * scale, extent + margin * scale)
    adjacency = torch.zeros(n, n)
    for edges, limit in ((b2b, n), (p2b, len(pins))):
        if edges.ndim != 2 or edges.shape[1] != 3:
            raise ValueError("Expected public edge triples")
        valid = edges[~(edges == -1).all(1)]
        if valid.numel() and ((valid[:, :2] < 0).any() or (valid[:, 0] >= limit).any()
                             or (valid[:, 1] >= n).any() or (valid[:, 2] < 0).any()
                             or (valid[:, :2] != valid[:, :2].long()).any()):
            raise ValueError("Invalid public edges")
    b2b = b2b[~(b2b == -1).all(1)]
    if len(b2b):
        a, b = b2b[:, 0].long(), b2b[:, 1].long()
        adjacency.index_put_((a, b), b2b[:, 2], accumulate=True)
        adjacency.index_put_((b, a), b2b[:, 2], accumulate=True)
    degree = adjacency.sum(1)
    # Exact per-module total weight and weighted center; no per-pin tokens.
    weight, center = torch.zeros(n), torch.zeros(n, 2)
    p2b = p2b[~(p2b == -1).all(1)]
    if len(p2b):
        pi, bi, w = p2b[:, 0].long(), p2b[:, 1].long(), p2b[:, 2]
        weight.index_add_(0, bi, w)
        center.index_add_(0, bi, pins[pi, :2] / side * w[:, None])
        center /= weight[:, None].clamp_min(1e-12)
    same = [(c[:, col, None] == c[None, :, col]) & (c[:, col, None] > 0) for col in (2, 3)]
    boundary = (c[:, 4].long()[:, None] & torch.tensor([1, 2, 4, 8]) != 0).float()
    wh = torch.where(locked[:, None], s[:, 2:] / side, 0.)
    xy = torch.where(pre[:, None], s[:, :2] / side, 0.)
    features = torch.cat((
        (areas / (side * side)).clamp_min(1e-12).log()[:, None],
        (effective / (side * side)).clamp_min(1e-12).log()[:, None],
        wh, xy, c[:, :2], boundary,
        degree.log1p()[:, None], weight.log1p()[:, None], center,
        *[x.sum(1, keepdim=True).float() / n for x in same],
        torch.from_numpy(module_palette(n)), torch.full((n, 1), scale / side)), dim=1)
    assert features.shape == (n, FEATURES)
    relation = torch.stack((adjacency.log1p(), same[0].float(), same[1].float()))
    padded = torch.zeros(MAX_MODULES, FEATURES)
    padded[:n] = features
    relations = torch.zeros(3, MAX_MODULES, MAX_MODULES)
    relations[:, :n, :n] = relation
    return {"features": padded, "relations": relations,
            "valid": torch.arange(MAX_MODULES) < n}, side


class RawImages(Dataset):
    def __init__(self, role, resolution, kappa=1.5, margin=.05, include_terminal_extent=False):
        if role not in ("train", "dev", "train_monitor500", "dev_monitor500"):
            raise PermissionError("E102 reads train/dev only; test remains sealed")
        self.reg = registry()
        self.records = self.reg["sets"][role]["records"]
        self.role, self.resolution, self.kappa, self.margin = role, resolution, kappa, margin
        self.include_terminal_extent = include_terminal_extent
        self.reader = None

    def __len__(self):
        return len(self.records)

    def raw(self, index):
        if self.reader is None:
            self.reader = FrozenMemberReader([r["member"] for r in self.records],
                "train" if self.role.startswith("train") else "dev",
                identity="E102.raw_reference_rgb.v1:" + self.reg["manifest_sha256"])
        # Direct masked raw reader: deliberately bypass mean_area_inputs and
        # ReferenceLayouts.__getitem__. Original fp_sol WHXY -> XYWH in reader.
        return self.reader.case(index)

    def __getitem__(self, index):
        raw = self.raw(index)
        conditioning, side = public_features(raw.solver_inputs, kappa=self.kappa, margin=self.margin,
            include_terminal_extent=self.include_terminal_extent)
        rects = np.asarray(raw.reference_positions)
        if (rects[:, :2] < -1e-6).any() or (rects[:, :2] + rects[:, 2:] > side + 1e-6).any():
            raise ValueError(f"Reference outside public canvas: {raw.case_id}; register canvas revision, do not crop")
        rgb = reference_rgb(rects, side, self.resolution)
        # Public conditioning is constructed before, and independently of, labels.
        target = torch.from_numpy(rgb).permute(2, 0, 1).contiguous() * 2 - 1
        member = self.records[index]["member"]
        key = int(hashlib.sha256(f"{member[0]}:{member[1]}".encode()).hexdigest()[:8], 16)
        return {**conditioning, "target": target, "key": key, "index": index}


def five_indices(dataset):
    """Metadata-only, spread module counts; no visual cherry-picking."""
    result = []
    for n in (21, 45, 70, 95, 120):
        candidates = [i for i in range(len(dataset)) if i not in result]
        result.append(min(candidates, key=lambda i: (abs(dataset.records[i]["n"] - n), i)))
    return result
