"""State and source identities shared by the semantic trainer and renderer."""
import hashlib
from pathlib import Path
import random
import zipfile

import numpy as np
import torch

from datasets.paths import ROOT


class TrainingStopped(Exception):
    pass


def get_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def set_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def snapshot(folder):
    package = Path(__file__).parent
    paths = list(package.glob("*.py")) + [package / "plan.json"]
    paths += [ROOT / p for p in ("experiments/E102_reference_image_generation/data.py",
        "experiments/E102_reference_image_generation/model.py", "experiments/E102_reference_image_generation/file_io.py",
        "experiments/E102_reference_image_generation/render.py", "visualization/raster_layouts.py",
        "visualization/semantic_images.py", "visualization/image_gallery.py", "datasets/paths.py",
        "datasets/worksets.py", "datasets/cases.py", "datasets/reference_layouts.py",
        "protocol/paths.json", "protocol/reference_workset_v5.json", "protocol/split_v1.json")]
    identity = {}
    with zipfile.ZipFile(folder / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            identity[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, name)
    return identity
