"""Saved best -> reference/generated RGB pairs, without geometry extraction."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from visualization.raster_layouts import comparison_sheet, save_rgb
from visualization.image_gallery import write_image_gallery
from .model import ReferenceImageGenerator, image_loss
from .prepare import CachedImages


def fixed_indices(dataset):
    indices = []
    for n in range(21, 121):
        matches = [i for i, r in enumerate(dataset.records) if r["n"] == n]
        if matches:
            indices.append(matches[0])
    return indices + [i for i in range(len(dataset)) if i not in indices][:100 - len(indices)]


@torch.inference_mode()
def publish(model, dataset, folder, *, device, provenance, batch_size=8):
    """Same fixed dev100 throughout; images do not select or schedule models."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    was_training = model.training
    model.eval()
    indices = fixed_indices(dataset)
    rows = []
    try:
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset + batch_size]
            items = [dataset[i] for i in selected]
            batch = {key: torch.stack([item[key] for item in items]).to(device)
                     for key in ("features", "relations", "valid", "target")}
            generated = model(batch)
            errors = image_loss(generated, batch["target"]).cpu().tolist()
            targets = (batch["target"].permute(0, 2, 3, 1).cpu().numpy() + 1) / 2
            predictions = (generated.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2
            for j, index in enumerate(selected):
                order = offset + j
                reference, output = f"{order:03d}_reference.png", f"{order:03d}_generated.png"
                save_rgb(folder / reference, targets[j])
                save_rgb(folder / output, predictions[j])
                comparison_sheet([targets[j], predictions[j]], ["Original reference", "Generated image"],
                    folder / f"{order:03d}_comparison.png", size=512,
                    caption=f"DEV | {dataset.records[index]['member']} | same canvas and module-ID colors")
                rows.append({"member": dataset.records[index]["member"], "mse": errors[j],
                    "reference": reference, "generated": output, "comparison": f"{order:03d}_comparison.png"})
        report = {"provenance": provenance, "image_mse": sum(r["mse"] for r in rows) / len(rows),
            "cases": rows, "geometry_extraction": False, "repair": False, "Q": None, "test_read": False}
        (folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_image_gallery(folder, rows, title="E102 · Reference / 生成图片",
            note=f"固定 dev100，同尺度与配色。{provenance['label']}。点击图片可查看原始512像素图。")
    finally:
        model.train(was_training)
    link = html.escape(folder.name + "/index.html", quote=True)
    note = html.escape(provenance["label"])
    content = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>E102 图片对比</title>'
        f'<meta http-equiv="refresh" content="0;url={link}"><body><p>{note}</p>'
        f'<a href="{link}">查看最新 reference / 生成图片对比</a></body></html>')
    temporary = folder.parent / "index.html.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(folder.parent / "index.html")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    plan = saved["plan"]
    dataset = CachedImages(args.cache, "dev_monitor500", plan)
    # Fixed one per count where available, then index-order fill to 100.
    indices = []
    for n in range(21, 121):
        matches = [i for i, r in enumerate(dataset.records) if r["n"] == n]
        if matches:
            indices.append(matches[0])
    indices += [i for i in range(len(dataset)) if i not in indices][:100 - len(indices)]
    args.out.mkdir(parents=True, exist_ok=False)
    model = ReferenceImageGenerator(**saved["model_config"]).to(args.device).eval()
    model.load_state_dict(saved["model"])
    rows = []
    with torch.inference_mode():
        for order, batch in enumerate(DataLoader(Subset(dataset, indices), batch_size=1)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            generated = model(batch)
            mse = float(image_loss(generated, batch["target"])[0])
            target, prediction = [(v[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
                                  for v in (batch["target"], generated)]
            save_rgb(args.out / f"{order:03d}_reference.png", target)
            save_rgb(args.out / f"{order:03d}_generated.png", prediction)
            index = indices[order]
            comparison_sheet([target, prediction], ["Original reference", "Generated image"],
                args.out / f"{order:03d}_comparison.png", size=512,
                caption=f"DEV | {dataset.records[index]['member']} | same canvas and module-ID colors")
            rows.append({"member": dataset.records[index]["member"], "mse": mse,
                         "comparison": f"{order:03d}_comparison.png"})
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "image_mse": sum(r["mse"] for r in rows) / len(rows), "cases": rows,
        "geometry_extraction": False, "repair": False, "Q": None, "test_read": False}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
