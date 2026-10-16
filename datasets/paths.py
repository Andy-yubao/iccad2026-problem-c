"""Resolve current physical data roots without rewriting the frozen split."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def dataset_paths(root=ROOT):
    root = Path(root).resolve()
    config = json.loads((root / "protocol/paths.json").read_text(encoding="utf-8"))
    paths = {name: (root / config[name]).resolve() for name in ("training_root", "public_validation_root")}
    for name, path in paths.items():
        if not path.is_relative_to(root) or not path.is_dir():
            raise FileNotFoundError(f"Missing/invalid local dataset {name}: {path}; no automatic download")
    return paths

def resolved_data_paths(root=ROOT):
    return {key: str(value) for key, value in dataset_paths(root).items()}

def current_data_relative(path, root=ROOT):
    """Interpret paths in immutable pre-migration installation records."""
    root = Path(root).resolve()
    roots = dataset_paths(root)
    old_roots = {"third_party/FloorSet/floorset_lite": roots["training_root"],
                 "third_party/FloorSet/LiteTensorDataTest": roots["public_validation_root"],
                 "downloads/datasets": root / "datasets/archives"}
    path = str(path).replace("\\", "/")
    for old, new in old_roots.items():
        if path == old or path.startswith(old + "/"):
            return new.relative_to(root).as_posix() + path[len(old):]
    return path

def train_index(root, split_path):
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    shards = split["shards"]["train"]
    if set(shards) & (set(split["shards"]["dev"]) | set(split["shards"]["local_test"])):
        raise ValueError("Split overlap")
    return dataset_paths(root)["training_root"], shards
