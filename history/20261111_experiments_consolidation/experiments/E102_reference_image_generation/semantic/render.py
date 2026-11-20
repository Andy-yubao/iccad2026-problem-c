"""Public-only all-MASK image generation on the unchanged dev100 panel."""
import json
from pathlib import Path

import torch

from ..file_io import replace_file
from ..render import fixed_indices
from visualization.semantic_images import ids_to_rgb
from visualization.raster_layouts import save_rgb, comparison_sheet
from visualization.image_gallery import write_image_gallery
from .model import MASK
from .precision import forward_image


def image_scores(prediction, target):
    b = len(target)
    offset = torch.arange(b, device=target.device)[:, None, None] * (121 * 121)
    confusion = torch.bincount((target * 121 + prediction + offset).flatten(), minlength=b * 121 * 121).reshape(b, 121, 121)
    intersection = confusion.diagonal(dim1=1, dim2=2).float()
    truth, guess = confusion.sum(2).float(), confusion.sum(1).float()
    iou = intersection / (truth + guess - intersection).clamp_min(1)
    present = truth[:, 1:] > 0
    return {"module_mean_IoU": (iou[:, 1:] * present).sum(1) / present.sum(1).clamp_min(1),
        "foreground_pixel_accuracy": intersection[:, 1:].sum(1) / truth[:, 1:].sum(1).clamp_min(1)}


@torch.inference_mode()
def publish(model, dataset, root, name, *, device, steps, monitor_id, label, should_stop, precision="FP32"):
    from .runtime import TrainingStopped
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=False)
    training = model.training
    model.eval()
    rows = []
    try:
        for order, index in enumerate(fixed_indices(dataset)):
            if reason := should_stop():
                raise TrainingStopped(reason)
            item = dataset[index]
            public = {k: item[k][None].to(device) for k in ("features", "relations", "valid")}
            # No target is passed as denoising state or public conditioning.
            tokens = torch.full((1, 512, 512), MASK, device=device, dtype=torch.long)
            prediction = forward_image(model, tokens, torch.ones(1, device=device), public, precision)["logits"].argmax(1)
            scores = {k: float(v[0]) for k, v in image_scores(prediction, item["labels"][None].to(device)).items()}
            reference = ids_to_rgb(item["labels"].numpy())
            generated = ids_to_rgb(prediction[0].cpu().numpy())
            a, b = f"{order:03d}_reference.png", f"{order:03d}_generated.png"
            save_rgb(folder / a, reference)
            save_rgb(folder / b, generated)
            comparison_sheet([reference, generated], ["Original reference", "Generated image"],
                folder / f"{order:03d}_comparison.png", caption=f"DEV | {dataset.records[index]['member']} | step {steps}")
            rows.append({"member": dataset.records[index]["member"], "reference": a, "generated": b,
                         "comparison": f"{order:03d}_comparison.png", **scores})
        report = {"label": label, "steps": steps, "monitor_id": monitor_id, "cases": rows,
                  "conditioning": "public only; full MASK state", "precision": precision, "test_read": False,
                  "module_mean_IoU": sum(r["module_mean_IoU"] for r in rows) / len(rows)}
        (folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_image_gallery(folder, rows, title="E102 新版 · 固定dev100图片对比", note=label + "。同一模块颜色和公开画布，左reference、右网络直接生成。")
        temporary = folder.parent / "index.html.tmp"
        temporary.write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=' + name + '/index.html"><a href="' + name + '/index.html">查看最新同例图片对比</a>', encoding="utf-8")
        replace_file(temporary, folder.parent / "index.html")
        return report
    finally:
        model.train(training)
