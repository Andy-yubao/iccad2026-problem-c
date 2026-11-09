"""Prepare the registered semantic image task, train/dev only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import SemanticImages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    for role in ("train", "train_monitor500", "dev_monitor500"):
        data = SemanticImages(role, 512)
        folder = args.out / role
        folder.mkdir()
        missing = []
        order = sorted(range(len(data)), key=lambda i: tuple(data.records[i]["member"]))
        for count, index in enumerate(order, 1):
            item = data[index]
            labels = item["labels"].numpy().astype(np.uint8)
            n = int(item["valid"].sum())
            invisible = sorted(set(range(1, n + 1)) - set(np.unique(labels).tolist()))
            if invisible:
                missing.append({"index": index, "member": data.records[index]["member"], "invisible_module_ids": invisible})
            np.savez_compressed(folder / f"{index:05d}.npz", labels=labels,
                **{key: item[key].numpy() for key in ("features", "relations", "valid")})
            if count % 500 == 0:
                print(f"{role}: {count}/{len(data)}", flush=True)
        metadata = {"task": "E102.semantic_masked_image.v2.1", "role": role,
            "resolution": 512, "canvas": "public_canvas_v2", "area_transform": "none",
            "registry_sha256": data.reg["manifest_sha256"], "records": data.records,
            "count": len(data), "members_with_invisible_modules": missing, "test_read": False}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.out / "complete.json").write_text(json.dumps({"status": "complete", "test_read": False}) + "\n")


if __name__ == "__main__":
    main()
