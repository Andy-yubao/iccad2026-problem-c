"""Explicit CPU image cache preparation; never reads test members."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import RawImages


def cache_identity(plan, dataset):
    return {"task": plan["task"], "resolution": plan["image"]["resolution"],
            "kappa": plan["image"]["kappa"], "margin": plan["image"]["margin"],
            "include_terminal_extent": plan["image"].get("include_terminal_extent", False),
            "representation": plan["image"]["representation"],
            "registry_sha256": dataset.reg["manifest_sha256"], "role": dataset.role,
            "members_sha256": hashlib.sha256(json.dumps([r["member"] for r in dataset.records]).encode()).hexdigest()}


class CachedImages(Dataset):
    def __init__(self, folder, role, plan):
        self.folder = Path(folder) / role
        raw = RawImages(role, plan["image"]["resolution"], plan["image"]["kappa"], plan["image"]["margin"],
            include_terminal_extent=plan["image"].get("include_terminal_extent", False))
        self.records = raw.records
        metadata = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["identity"] != cache_identity(plan, raw) or metadata["count"] != len(raw):
            raise ValueError("Cache identity differs from the registered image task")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with np.load(self.folder / f"{index:05d}.npz", allow_pickle=False) as item:
            values = {key: torch.from_numpy(item[key].copy()) for key in ("features", "relations", "valid")}
            values["target"] = torch.from_numpy(item["rgb"].copy()).permute(2, 0, 1).float() / 127.5 - 1
        return {**values, "index": index}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for role in ("train", "train_monitor500", "dev_monitor500"):
        raw = RawImages(role, plan["image"]["resolution"], plan["image"]["kappa"], plan["image"]["margin"],
            include_terminal_extent=plan["image"].get("include_terminal_extent", False))
        folder = args.out / role
        folder.mkdir()
        # Group disk reads only during preparation; training retains original
        # member indices and its separately saved random permutation.
        order = sorted(range(len(raw)), key=lambda i: tuple(raw.records[i]["member"]))
        for count, index in enumerate(order, 1):
            item = raw[index]
            rgb = ((item["target"].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
            np.savez_compressed(folder / f"{index:05d}.npz", rgb=rgb,
                **{key: item[key].numpy() for key in ("features", "relations", "valid")})
            if count % 500 == 0:
                print(f"{role}: {count}/{len(raw)}", flush=True)
        metadata = {"identity": cache_identity(plan, raw), "count": len(raw), "test_read": False}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
