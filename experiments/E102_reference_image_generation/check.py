"""One real CPU optimizer update on the same five train examples, nothing more."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .data import RawImages, five_indices
from .model import ReferenceImageGenerator, image_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(__file__).with_name("plan.json").read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    torch.manual_seed(plan["training"]["seed"])
    ds = RawImages("train_monitor500", plan["image"]["resolution"], plan["image"]["kappa"], plan["image"]["margin"])
    indices = five_indices(ds)
    # Same 8-bit rounding as formal prepared targets, without preparing a dataset.
    samples = []
    for i in indices:
        sample = ds[i]
        sample["target"] = ((sample["target"] + 1) * 127.5).round() / 127.5 - 1
        samples.append(sample)
    batch = default_collate(samples)
    assert all(torch.isfinite(batch[k]).all() for k in ("features", "relations", "target"))
    model = ReferenceImageGenerator(**plan["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["training"]["lr"], weight_decay=plan["training"]["weight_decay"])
    before = model.output.weight.detach().clone()
    begin = time.perf_counter()
    prediction = model(batch)
    assert prediction.shape == batch["target"].shape == (5, 3, 512, 512)
    loss = image_loss(prediction, batch["target"]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    finite = bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all())
    changed = float((model.output.weight.detach() - before).abs().max())
    assert finite and changed > 0
    report = {"status": "passed", "actual_optimizer_updates": 1, "device": "CPU",
        "resolution": 512, "check_batch": 5, "formal_batch": 8,
        "members": [ds.records[i]["member"] for i in indices],
        "registry_sha256": ds.reg["manifest_sha256"],
        "parameters": sum(p.numel() for p in model.parameters()), "initial_mse": float(loss.detach()),
        "gradient_norm": float(grad), "max_output_weight_update": changed,
        "finite_parameters": finite, "update_check_seconds": time.perf_counter() - begin,
        "formal_training_started": False, "quality_claim": False,
        "area_transform": "none", "geometry_extraction": False, "repair": False,
        "Q_computed": False, "test_read": False, "E101_GPU_used": False}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
