"""Five-case causal audit of the archived official baseline, without re-running SA.

Reuses five saved solver outputs and compares the dataset reference and a crude
input-only legal layout. Reference labels are used only for checker diagnostics;
the legal control sees only the same masked fixed/preplaced specs as solve().
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.paths import dataset_paths, current_data_relative

CASE_IDS = [0, 19, 39, 69, 99]  # 21, 40, 60, 90, 120 blocks


from scoring.checks import independent_hard_checks

def input_only_legal_control(areas, constraints, allowed_specs):
    """Preserve preplaced, then place all movable blocks in a row to their right.

    This is a diagnostic feasibility witness, not a competitive floorplanner.
    No free-block target position/shape or ground-truth metrics is consulted.
    """
    n = len(areas)
    positions = [None] * n
    for i in range(n):
        if bool(constraints[i, 1]):
            positions[i] = [float(v) for v in allowed_specs[i]]
    cursor = max([p[0] + p[2] for p in positions if p is not None] + [0.0]) + 1.0
    for i in range(n):
        if positions[i] is not None:
            continue
        if bool(constraints[i, 0]):
            w, h = map(float, allowed_specs[i][2:])
        else:
            w = h = math.sqrt(float(areas[i]))
        positions[i] = [cursor, 0.0, w, h]
        cursor += w + 1.0
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    sys.path.insert(0, str(ROOT / 'third_party/FloorSet/iccad2026contest'))
    import numpy as np
    import torch
    import iccad2026_evaluate as official
    torch.set_num_threads(1)
    run = ROOT / 'experiments/baseline/runs/full100_clean_cu128_seed0/run'
    saved = json.loads((run / 'results.json').read_text())
    source_manifest = json.loads((ROOT / 'third_party/upstream_manifest.json').read_text())
    source_checks = []
    for record in source_manifest['files']:
        if record['path'].endswith(('/optimizer_template.py', '/iccad2026_evaluate.py', '/lite_dataset_test.py')):
            actual = hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest()
            source_checks.append({'path': record['path'], 'sha256_matches_archive': actual == record['sha256']})
    assert len(source_checks) == 3 and all(r['sha256_matches_archive'] for r in source_checks)
    validation_manifest = json.loads((ROOT / 'datasets/validation_installation.json').read_text())
    known_data = {current_data_relative(r['path'], ROOT): r for r in validation_manifest['files']}
    evaluator = official.ContestEvaluator(str(dataset_paths(ROOT)['public_validation_root'].parent), verbose=False)
    evaluator._load_dataset()
    cases = []
    started = time.perf_counter()
    for idx in CASE_IDS:
        row = next(r for r in saved['test_results'] if r['test_id'] == idx)
        sample = evaluator.dataset[idx]
        areas, b2b, p2b, pins, constraints = sample['input']
        baseline, target = evaluator._extract_baseline(idx, sample['label'], b2b, p2b, pins, row['block_count'])
        n = row['block_count']
        # Derive XYWH independently from the raw label polygons.
        decoded = []
        rectangle_labels = []
        for polygon in sample['label'][0]:
            vertices = polygon.numpy()
            vertices = vertices[np.all(vertices != -1, axis=1)]
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            decoded.append([float(lower[0]), float(lower[1]), float(upper[0] - lower[0]), float(upper[1] - lower[1])])
            rectangle_labels.append(len(set(map(tuple, vertices))) == 4)
        assert np.allclose(decoded, target, atol=0, rtol=0)
        allowed = torch.full((n, 4), -1.0)
        for i in range(n):
            if bool(constraints[i, 1]):
                allowed[i] = torch.tensor(target[i])
            elif bool(constraints[i, 0]):
                allowed[i, 2:] = torch.tensor(target[i][2:])
        # Confirm the saved input/label pair has the official loader's expected size/index.
        files = []
        for path in [evaluator.dataset.all_input_files[idx], evaluator.dataset.all_label_files[idx]]:
            path = Path(path)
            rel = path.relative_to(ROOT).as_posix()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({'path': rel, 'sha256_verified': actual == known_data[rel]['sha256']})
        assert all(r['sha256_verified'] for r in files)
        control = input_only_legal_control(areas, constraints, allowed)
        variants = {}
        for name, positions in [('saved_official_output', row['positions']), ('reference_label', decoded), ('input_only_legal_control', control)]:
            metrics = official.evaluate_solution({'positions': positions, 'runtime': 1.0}, baseline,
                constraints, b2b, p2b, pins, areas, target, median_runtime=1.0)
            independent = independent_hard_checks(positions, areas, constraints, decoded)
            assert independent['is_feasible'] == metrics.is_feasible
            assert independent['dimension_or_location_violation_count'] == metrics.dimension_violations
            variants[name] = {'metrics': asdict(metrics), 'independent_hard_checks': independent}
        original = variants['saved_official_output']['metrics']
        assert original['cost'] == row['cost'] and original['is_feasible'] == row['is_feasible']
        bad_examples = []
        for i in sorted(set(variants['saved_official_output']['independent_hard_checks']['fixed_or_preplaced_shape_violations']) |
                        set(variants['saved_official_output']['independent_hard_checks']['preplaced_location_violations'])):
            bad_examples.append({'block_id': i, 'fixed': bool(constraints[i, 0]), 'preplaced': bool(constraints[i, 1]),
                                 'spec_xywh': decoded[i], 'output_xywh': row['positions'][i]})
        # Independent vectorized connectivity calculation, including pin/block index order.
        xywh = np.array(decoded, dtype=np.float64)
        centers = xywh[:, :2] + xywh[:, 2:] / 2
        be, pe = b2b.numpy(), p2b.numpy()
        b2b_wl = float((np.abs(centers[be[:, 0].astype(int)] - centers[be[:, 1].astype(int)]).sum(axis=1) * be[:, 2]).sum())
        p2b_wl = float((np.abs(pins.numpy()[pe[:, 0].astype(int)] - centers[pe[:, 1].astype(int)]).sum(axis=1) * pe[:, 2]).sum())
        reference = variants['reference_label']['metrics']
        assert abs(b2b_wl - reference['hpwl_b2b']) < 1e-6
        assert abs(p2b_wl - reference['hpwl_p2b']) < 1e-6
        case = {'test_id': idx, 'blocks': n, 'input_label_hash_checks': files,
                'decoded_target_matches_official_exactly': True, 'all_labels_have_four_unique_vertices': all(rectangle_labels),
                'constraint_counts': {'fixed': int((constraints[:,0] != 0).sum()), 'preplaced': int((constraints[:,1] != 0).sum())},
                'allowed_specs_xywh': allowed.tolist(), 'input_only_control_positions': control,
                'hpwl_independent_calculation_matches': True, 'variants': variants, 'violating_block_examples': bad_examples}
        cases.append(case)
        print(f"case {idx:02} n={n}: official hard=({original['overlap_violations']},{original['area_violations']},{original['dimension_violations']}); reference feasible={reference['is_feasible']}; legal control feasible={variants['input_only_legal_control']['metrics']['is_feasible']}", flush=True)
    summary = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'case_ids': CASE_IDS,
               'new_SA_runs': 0, 'saved_SA_outputs_reused': 5, 'scope': 'Five selected validation cases only',
               'python_executable': sys.executable, 'torch_version': torch.__version__,
               'source_integrity_checks': source_checks, 'wall_seconds': time.perf_counter() - started,
               'official_outputs_feasible': sum(c['variants']['saved_official_output']['metrics']['is_feasible'] for c in cases),
               'reference_layouts_feasible': sum(c['variants']['reference_label']['metrics']['is_feasible'] for c in cases),
               'input_only_controls_feasible': sum(c['variants']['input_only_legal_control']['metrics']['is_feasible'] for c in cases),
               'cases': cases}
    (out / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Audit saved to', out / 'audit.json')


if __name__ == '__main__':
    main()
