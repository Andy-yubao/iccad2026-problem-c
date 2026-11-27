"""Module-ID image targets and palette display; never extract rectangles."""
import numpy as np
from .raster_layouts import module_palette


def reference_id_image(rectangles, side, resolution):
    """Pixel-center labels: 0=background, module i=i+1. No fitted bbox."""
    image = np.zeros((resolution, resolution), dtype=np.uint8)
    coordinates = (np.arange(resolution) + .5) * side / resolution
    for i, (x, y, w, h) in enumerate(np.asarray(rectangles, dtype=np.float64)):
        cols = np.flatnonzero((coordinates >= x) & (coordinates < x + w))
        rows = np.flatnonzero((coordinates >= y) & (coordinates < y + h))
        if len(cols) and len(rows):
            image[np.ix_(resolution - 1 - rows, cols)] = i + 1
    return image


def ids_to_rgb(ids, count=120):
    palette = np.concatenate((np.ones((1, 3), dtype=np.float32), module_palette(count)))
    ids = np.asarray(ids)
    if ids.min() < 0 or ids.max() > count:
        raise ValueError("Invalid image-label ID")
    return palette[ids]
