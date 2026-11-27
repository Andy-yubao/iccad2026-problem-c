"""Load the archived, unmodified official evaluator."""
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def official_evaluator():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    contest = str(ROOT / "third_party/FloorSet/iccad2026contest")
    if contest not in sys.path:
        sys.path.insert(0, contest)
    official = importlib.import_module("iccad2026_evaluate")
    if not official.SHAPELY_AVAILABLE:
        raise RuntimeError("Shapely is required; otherwise grouping violations would be silently omitted")
    return official
