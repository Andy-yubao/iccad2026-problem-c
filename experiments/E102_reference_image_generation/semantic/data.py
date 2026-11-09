"""Original reference label images, public-only conditioning, frozen members."""
from pathlib import Path
import json

import numpy as np
import torch

from ..data import RawImages, public_features, five_indices
from visualization.semantic_images import reference_id_image


class SemanticImages(RawImages):
    def __init__(self, role, resolution=512):
        super().__init__(role, resolution, kappa=1.5, margin=.05, include_terminal_extent=True)

    def __getitem__(self, index):
        raw = self.raw(index)
        condition, side = public_features(raw.solver_inputs, kappa=self.kappa, margin=self.margin,
                                         include_terminal_extent=True)
        rects = np.asarray(raw.reference_positions)
        if (rects[:, :2] < -1e-6).any() or (rects[:, :2] + rects[:, 2:] > side + 1e-6).any():
            raise ValueError("Reference exceeds registered public canvas")
        labels = torch.from_numpy(reference_id_image(rects, side, self.resolution)).long()
        return {**condition, "labels": labels, "index": index}


class CachedSemanticImages(torch.utils.data.Dataset):
    def __init__(self, root, role):
        if role not in ("train", "train_monitor500", "dev_monitor500"):
            raise PermissionError("Only frozen train/dev are allowed")
        self.root = Path(root) / role
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["task"] != "E102.semantic_masked_image.v2.1":
            raise ValueError("Wrong image task")
        expected = SemanticImages(role, 512)
        if (self.metadata["registry_sha256"] != expected.reg["manifest_sha256"]
                or self.metadata["records"] != expected.records
                or self.metadata["count"] != len(expected)
                or self.metadata["resolution"] != 512):
            raise ValueError("Cache differs from the registered frozen members or image task")
        self.records = self.metadata["records"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with np.load(self.root / f"{index:05d}.npz", allow_pickle=False) as item:
            result = {key: torch.from_numpy(item[key].copy()) for key in ("features", "relations", "valid")}
            result["labels"] = torch.from_numpy(item["labels"].copy()).long()
        return {**result, "index": index}
