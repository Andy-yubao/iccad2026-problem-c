"""Plain reference-image targets. No scoring, geometry recovery, or repair."""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def module_palette(count=120):
    """Stable module-ID colors; same ID has the same color in every image."""
    return np.array([colorsys.hsv_to_rgb((i * .61803398875) % 1,
                     .48 + .14 * (i % 3), .92 - .10 * (i % 2))
                     for i in range(count)], dtype=np.float32)


def reference_rgb(rectangles, side, resolution):
    """Exact pixel-area coverage of solid rectangles on a white square.

    XYWH uses bottom-left origin; images use top-left origin. No text, hatches,
    artificial gaps, border inflation, bbox fitting, or min-size enlargement.
    Coverage antialiasing keeps fractional-pixel slivers visible. The reference
    is an original rectangular tiling; shared pixel contributions add rather
    than alpha-occlude each other. Nothing is decoded or repaired.
    """
    rectangles = np.asarray(rectangles, dtype=np.float64)
    colors = module_palette(len(rectangles))
    image = np.ones((resolution, resolution, 3), dtype=np.float32)
    for color, (x, y, w, h) in zip(colors, rectangles):
        x0, x1 = x / side * resolution, (x + w) / side * resolution
        y0, y1 = resolution - (y + h) / side * resolution, resolution - y / side * resolution
        left, right = max(0, int(np.floor(x0))), min(resolution, int(np.ceil(x1)))
        top, bottom = max(0, int(np.floor(y0))), min(resolution, int(np.ceil(y1)))
        if right <= left or bottom <= top:
            continue
        xs, ys = np.arange(left, right), np.arange(top, bottom)
        cx = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
        cy = np.clip(np.minimum(ys + 1, y1) - np.maximum(ys, y0), 0, 1)
        coverage = (cy[:, None] * cx[None, :])[..., None]
        image[top:bottom, left:right] += (color - 1) * coverage
    return np.clip(image, 0, 1)


def save_rgb(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + .5)).save(path)


def comparison_sheet(images, titles, destination, *, caption="", size=512):
    """Nearest-neighbor equal-size display preserves actual raster differences."""
    width = len(images) * (size + 20) + 20
    sheet = Image.new("RGB", (width, size + 90), "#f1f5f9")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), caption, fill="#0f172a")
    for i, (array, title) in enumerate(zip(images, titles)):
        tile = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + .5))
        tile = tile.resize((size, size), Image.Resampling.NEAREST)
        x = 20 + i * (size + 20)
        draw.text((x, 38), title, fill="#0f172a")
        sheet.paste(tile, (x, 64))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
