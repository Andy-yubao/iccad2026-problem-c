"""Frozen dev/public loading and masked solver inputs. local_test is sealed."""
from dataclasses import dataclass
import math
from pathlib import Path
from datasets.paths import ROOT, dataset_paths
from scoring.official import official_evaluator

PUBLIC_CASE_IDS = [0, 19, 39, 69, 99]

@dataclass
class ResearchCase:
    case_id: str
    source: dict
    solver_inputs: tuple
    reference_positions: list
    baseline_metrics: dict

    @property
    def block_count(self):
        return self.solver_inputs[0]

    def input_summary(self):
        constraints = self.solver_inputs[5]
        return {
            "block_count": self.block_count,
            "fixed_blocks": int((constraints[:, 0] != 0).sum()),
            "preplaced_blocks": int((constraints[:, 1] != 0).sum()),
            "boundary_blocks": int((constraints[:, 4] != 0).sum()),
            "grouped_blocks": int((constraints[:, 3] != 0).sum()),
            "mib_blocks": int((constraints[:, 2] != 0).sum()),
        }


def allowed_target_positions(reference_positions, constraints):
    """Expose XYWH for preplaced, WH for fixed, and nothing for free blocks."""
    import torch
    n = len(reference_positions)
    allowed = torch.full((n, 4), -1.0, dtype=torch.float32)
    for i in range(n):
        if bool(constraints[i, 1]):
            allowed[i] = torch.tensor(reference_positions[i], dtype=allowed.dtype)
        elif bool(constraints[i, 0]):
            allowed[i, 2:] = torch.tensor(reference_positions[i][2:], dtype=allowed.dtype)
    return allowed


def _build_case(case_id, source, ordinary_inputs, reference, metrics):
    areas, b2b, p2b, pins, constraints = [value.detach().cpu().clone() for value in ordinary_inputs]
    n = int((areas != -1).sum())
    if n != len(areas) or len(reference) != n:
        raise ValueError(f"{case_id}: expected unpadded inputs and exactly {n} reference rectangles")
    if constraints.ndim != 2 or constraints.shape[1] < 5:
        raise ValueError(f"{case_id}: unsupported constraint schema {tuple(constraints.shape)}")
    reference = [[float(v) for v in row] for row in reference]
    if any(len(row) != 4 or not all(math.isfinite(v) for v in row) or min(row[2:]) <= 0
           for row in reference):
        raise ValueError(f"{case_id}: malformed reference geometry")
    official = official_evaluator()
    baseline = {
        "hpwl_baseline": official.calculate_hpwl_b2b(reference, b2b)
                         + official.calculate_hpwl_p2b(reference, p2b, pins),
        "area_baseline": official.calculate_bbox_area(reference),
    }
    # Match ContestEvaluator._extract_baseline's preference for stored metrics.
    if metrics is not None and len(metrics) >= 8:
        b2b_wl = official.calculate_hpwl_b2b(reference, b2b)
        p2b_wl = official.calculate_hpwl_p2b(reference, p2b, pins)
        if float(metrics[0]) > 0:
            baseline["area_baseline"] = float(metrics[0])
        if float(metrics[-2]) > 0:
            b2b_wl = float(metrics[-2])
        if float(metrics[-1]) >= 0:
            p2b_wl = float(metrics[-1])
        baseline["hpwl_baseline"] = b2b_wl + p2b_wl
    specs = allowed_target_positions(reference, constraints)
    return ResearchCase(case_id, source, (n, areas, b2b, p2b, pins, constraints, specs),
                        reference, baseline)


def load_frozen_cases(panel="dev", root=ROOT, limit=5):
    """Read only the five frozen dev samples or five public compatibility cases.

    Training fp_sol uses WHXY; public labels are polygons.  Loading follows the
    official raw file schemas directly and never calls its auto-downloader.
    """
    import json
    import torch
    root = Path(root).resolve()
    if not 1 <= limit <= 5:
        raise ValueError("Frozen lightweight panel limit must be 1..5")
    split = json.loads((root / "protocol/split_v1.json").read_text(encoding="utf-8"))
    cases = []
    if panel == "dev":
        selected = split["dev_smoke_panel"]
        if len(selected) != 5 or any(entry["shard"] not in split["shards"]["dev"] for entry in selected):
            raise ValueError("Frozen dev panel must contain exactly five development samples")
        data_root = dataset_paths(root)["training_root"]
        for entry in selected[:limit]:
            path = (data_root / entry["shard"]).resolve()
            if not path.is_relative_to(data_root):
                raise ValueError("Invalid dev shard path")
            raw = torch.load(path, map_location="cpu", weights_only=False)
            idx = int(entry["sample_index"])
            block_features = raw[0][idx]
            ordinary = (block_features[:, 0], raw[1][idx], raw[2][idx], raw[3][idx], block_features[:, 1:])
            # Official training schema: width, height, x, y -> x, y, width, height.
            reference = raw[5][idx][:, [2, 3, 0, 1]].tolist()
            source = {"split": "dev", "shard": entry["shard"], "sample_index": idx,
                      "paths": [path.relative_to(root).as_posix()]}
            cases.append(_build_case(f"dev:{entry['shard']}:{idx}", source, ordinary, reference, raw[6][idx]))
    elif panel == "public":
        if split["public_compatibility_case_ids"] != PUBLIC_CASE_IDS:
            raise ValueError("Unexpected public compatibility panel; protocol review required")
        for idx in PUBLIC_CASE_IDS[:limit]:
            folder = dataset_paths(root)["public_validation_root"] / f"config_{21 + idx}"
            input_path, label_path = folder / "litedata_1.pth", folder / "litelabel_1.pth"
            raw_input = torch.load(input_path, map_location="cpu", weights_only=False)[0]
            metrics, polygons = torch.load(label_path, map_location="cpu", weights_only=False)[0]
            ordinary = (raw_input[0][:, 0], raw_input[1], raw_input[2], raw_input[3], raw_input[0][:, 1:])
            reference = []
            for polygon in polygons:
                valid = polygon[polygon[:, 0] != -1]
                lower, upper = valid.min(dim=0).values, valid.max(dim=0).values
                reference.append([*lower.tolist(), *(upper - lower).tolist()])
            source = {"split": "public_compatibility", "public_case_id": idx,
                      "paths": [p.relative_to(root).as_posix() for p in (input_path, label_path)]}
            cases.append(_build_case(f"public:{idx}", source, ordinary, reference, metrics))
    else:
        raise ValueError("Only dev and public panels are available; local_test is sealed")
    return cases
