"""Five-train-image resolution observation; no raster-to-rectangle path."""
import argparse
import json
from pathlib import Path

import numpy as np

from visualization.raster_layouts import reference_rgb, save_rgb, comparison_sheet
from .data import RawImages, five_indices, public_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kappa", type=float, default=1.5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    dataset = RawImages("train_monitor500", 256, kappa=args.kappa)
    resolutions = (128, 256, 512, 1024)
    rows = []
    for order, index in enumerate(five_indices(dataset)):
        raw = dataset.raw(index)
        _, side = public_features(raw.solver_inputs, kappa=args.kappa, margin=.05)
        rects = np.asarray(raw.reference_positions)
        small = np.argsort(rects[:, 2:].min(1), kind="stable")[:5]
        # The same physical crop at each resolution; crop is display only.
        focus = rects[small[0]]
        center = focus[:2] + focus[2:] / 2
        half = side * .08
        low = np.maximum(0, center - half)
        high = np.minimum(side, center + half)
        images, crops, stats = [], [], {}
        for resolution in resolutions:
            image = reference_rgb(rects, side, resolution)
            images.append(image)
            prefix = f"case_{order:02d}_{resolution}"
            save_rgb(args.out / f"{prefix}.png", image)
            pixel_min = rects[:, 2:].min(1) / side * resolution
            left, right = int(low[0] / side * resolution), int(np.ceil(high[0] / side * resolution))
            top = int((1 - high[1] / side) * resolution)
            bottom = int(np.ceil((1 - low[1] / side) * resolution))
            crop = image[max(0, top):min(resolution, bottom), max(0, left):min(resolution, right)]
            crops.append(crop)
            stats[str(resolution)] = {
                "min_short_side_pixels": float(pixel_min.min()),
                "blocks_shorter_than_1px": int((pixel_min < 1).sum()),
                "blocks_shorter_than_2px": int((pixel_min < 2).sum()),
                "blocks_shorter_than_4px": int((pixel_min < 4).sum()),
                "smallest_five_ids": small.tolist(),
                "smallest_five_short_side_pixels": pixel_min[small].tolist()}
        caption = f"TRAIN ONLY | {raw.case_id} | n={len(rects)} | fixed public canvas | no repair"
        comparison_sheet(images, [str(r) + " x " + str(r) for r in resolutions],
                         args.out / f"case_{order:02d}_comparison.png", caption=caption, size=512)
        comparison_sheet(crops, [str(r) + ": smallest-block region" for r in resolutions],
                         args.out / f"case_{order:02d}_detail.png", caption=caption, size=384)
        overflow = float(max(0., (rects[:, :2] + rects[:, 2:] - side).max()))
        rows.append({"case_id": raw.case_id, "member": dataset.records[index]["member"],
                     "n": len(rects), "canvas_side": side, "overflow": overflow,
                     "negative_coordinate": bool((rects[:, :2] < -1e-6).any()), "resolutions": stats})
    report = {"experiment": "E102", "purpose": "Small-block visual fidelity, not inverse geometry",
              "panel": "five train_monitor500 members selected by module-count metadata",
              "registry_sha256": dataset.reg["manifest_sha256"], "kappa": args.kappa,
              "margin": .05, "antialias": "analytic rectangle pixel coverage",
              "area_transform": "none", "test_read": False, "cases": rows}
    (args.out / "resolution_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
