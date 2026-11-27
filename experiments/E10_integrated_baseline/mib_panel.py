"""Paired baseline diagnosis on a public-input, mean-area MIB dev panel."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def mean_area_inputs(inputs):
    """Return a strict common-shape witness, or reject immutable shape conflicts."""
    import numpy as np
    values = tuple(v.clone() if hasattr(v, 'clone') else v for v in inputs)
    _, areas, _, _, _, constraints, specs = values
    c, s = np.asarray(constraints), np.asarray(specs)
    locked = c[:, :2].any(1)
    changes = []
    for group in sorted(set(c[:, 2]) - {0}):
        ids = np.flatnonzero(c[:, 2] == group)
        old = areas[ids].tolist()
        common = float(areas[ids].double().mean())
        areas[ids] = common
        common = float(areas[ids[0]])  # actual serialized float32 value
        rigid = ids[locked[ids]]
        if len(rigid):
            wh = s[rigid[0], 2:]
            if not np.allclose(s[rigid, 2:], wh, rtol=0, atol=1e-10):
                return None, 'locked_dimensions_disagree'
            if abs(float(wh.prod()) / common - 1) >= .009999:
                return None, 'mean_area_incompatible_with_locked_shape'
        else:
            wh = np.full(2, np.sqrt(common))
        changes.append({'group': int(group), 'blocks': ids.tolist(),
                        'old_areas': old, 'new_area': common,
                        'common_shape_witness': wh.tolist(),
                        'locked_blocks': rigid.tolist()})
    return values, changes


def write_json(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare(output):
    import torch
    from datasets.worksets import LayoutReader, workset_manifest
    from datasets.cases import _build_case
    from datasets.paths import dataset_paths
    manifest = workset_manifest()
    quality = LayoutReader(manifest, 'dev_quality')
    dev = LayoutReader(manifest, 'dev')
    lookup = {pair: i for i, pair in enumerate(dev.members)}
    chosen, audit, used = [], [], set()
    # One per worker when possible; quality member first, then frozen dev order.
    for worker in range(100):
        candidates = [p for p in dev.members if p[0].startswith(f'worker_{worker}/')]
        preferred = quality.members[worker]
        assert preferred[0].startswith(f'worker_{worker}/')
        candidates = [preferred] + [p for p in candidates if p != preferred]
        for pair in candidates:
            raw = dev.case(lookup[pair])
            inputs, changes = mean_area_inputs(raw.solver_inputs)
            audit.append({'case_id': raw.case_id, 'accepted': inputs is not None,
                          'reason': 'compatible' if inputs is not None else changes})
            if inputs is None:
                continue
            case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6],
                               raw.reference_positions, raw.stored_metrics)
            chosen.append({'case': case, 'derived_inputs': inputs, 'changes': changes})
            used.add(pair)
            break
    # Exhausted workers cannot satisfy immutable dimensions under the mean rule.
    # Complete the panel in frozen dev order; disclose the resulting imbalance.
    attempted = {a['case_id'] for a in audit}
    for pair in dev.members:
        if len(chosen) == 100: break
        if pair in used: continue
        case_id = f'dev:{pair[0]}:{pair[1]}'
        if case_id in attempted: continue
        raw = dev.case(lookup[pair])
        inputs, changes = mean_area_inputs(raw.solver_inputs)
        audit.append({'case_id': raw.case_id, 'accepted': inputs is not None,
                      'reason': 'compatible' if inputs is not None else changes})
        if inputs is None: continue
        case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6],
                           raw.reference_positions, raw.stored_metrics)
        chosen.append({'case': case, 'derived_inputs': inputs, 'changes': changes})
        used.add(pair)
    assert len(chosen) == 100
    output.mkdir(parents=True, exist_ok=False)
    torch.save(chosen, output / 'dataset.pt')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(output / 'panel.json', {'selection': 'one per worker when possible, dev_quality first then frozen dev order; first mean-area-compatible member; fill exhausted-worker deficit in frozen dev order',
        'selected_count': len(chosen), 'examined_count': len(audit),
        'worker_counts': dict(Counter(e['case'].source['shard'].split('/')[0] for e in chosen)),
        'retained_dev_quality_members': sum((e['case'].source['shard'], e['case'].source['sample_index']) in quality.members for e in chosen),
        'rejections': dict(Counter(a['reason'] for a in audit if not a['accepted'])),
        'members': [{'case_id': e['case'].case_id, 'source': e['case'].source, 'changes': e['changes']} for e in chosen],
        'selection_audit': audit, 'local_test_read': False,
        'scope': 'Exact local MIB/area/locked-shape compatibility only; does not prove joint boundary/grouping/geometric feasibility.'})
    write_json(output / 'config.json', {'command': sys.argv, 'seed': 0, 'threads': 2, 'device': 'cpu',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
        'split_sha256': digest(ROOT / 'protocol/split_v1.json'),
        'workset_sha256': digest(ROOT / 'protocol/workset_v2.json'),
        'dataset_sha256': digest(output / 'dataset.pt'),
        'data_paths': {k: str(v) for k, v in dataset_paths().items()},
        'seconds_per_case_target': 5, 'variants': ['original', 'mean_area'],
        'budget': '100 paired cases, 200 solves, frozen baseline; no training or tuning',
        'reference_policy': 'Keep original stored HPWL/bbox denominators as fixed diagnostic anchors. Original free-block reference is not a feasible reference for modified areas; this is not a calibrated synthetic contest score.',
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in Path(__file__).parent.rglob('*.py')},
        'local_test_read': False})
    # Tiny exact source snapshot makes uncommitted code recoverable.
    import zipfile
    with zipfile.ZipFile(output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in Path(__file__).parent.rglob('*.py'):
            archive.write(p, p.relative_to(ROOT))
    print('PREPARED', len(chosen), 'examined', len(audit), flush=True)


def run(output):
    import torch
    import numpy as np
    from copy import copy
    from experiments.E10_integrated_baseline.pipeline import Baseline, CHECKPOINT_SHA256
    from scoring.evaluation import evaluate_case
    from scoring.official import official_evaluator
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    entries = torch.load(output / 'dataset.pt', weights_only=False)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((output / 'dataset.pt').read_bytes()).hexdigest() == config['dataset_sha256']
    config.update(checkpoint_sha256=CHECKPOINT_SHA256, torch_version=torch.__version__,
                  numpy_version=np.__version__, run_command=sys.argv)
    write_json(output / 'config.json', config)
    baseline = Baseline('cpu')
    rows = []
    if (output / 'results.json').exists():
        rows = json.loads((output / 'results.json').read_text(encoding='utf-8'))['cases']
    start = time.perf_counter()
    for entry in entries[len(rows):]:
        original = entry['case']
        row = {'case_id': original.case_id, 'block_count': original.block_count, 'variants': {}}
        for variant in ['original', 'mean_area']:
            case = copy(original)
            if variant == 'mean_area': case.solver_inputs = entry['derived_inputs']
            begin = time.perf_counter()
            try:
                positions, stats = baseline.solve(case.solver_inputs)
                result = evaluate_case(case, positions, stats['end_to_end_seconds'])
                stats['improvements'] = [{k: v for k, v in e.items() if k != 'positions'}
                                         for e in stats.get('improvements', [])]
                row['variants'][variant] = {'result': result, 'stats': stats}
            except Exception as exc:
                row['variants'][variant] = {'error': repr(exc), 'seconds': time.perf_counter() - begin,
                    'result': {'metrics': {'is_feasible': False, 'cost': 10.}}}
        rows.append(row)
        summaries = {}
        for variant in ['original', 'mean_area']:
            records = [r['variants'][variant] for r in rows]
            metrics = [r['result']['metrics'] for r in records]
            costs = [m['cost'] for m in metrics]
            summaries[variant] = {'Q_mean': float(np.mean(costs)), 'Q_weighted': official_evaluator().compute_total_score(costs, [r['block_count'] for r in rows]),
                'Q_max': max(costs), 'Q_median': float(np.median(costs)), 'Q_p90': float(np.quantile(costs, .9)),
                'feasible': sum(m['is_feasible'] for m in metrics), 'errors': sum('error' in r for r in records),
                'mib_zero_cases': sum(m.get('mib_violations') == 0 for m in metrics),
                'mean_inference_plus_postprocess_seconds': float(np.mean([r.get('stats', {}).get('end_to_end_seconds', r.get('seconds', 0)) for r in records])),
                **{f'mean_{key}': float(np.mean([m[key] for m in metrics if key in m])) for key in
                   ['hpwl_gap', 'area_gap', 'boundary_violations', 'grouping_violations', 'mib_violations']}}
        report = {'status': 'complete' if len(rows) == 100 else 'running', 'completed_pairs': len(rows),
                  'summary': summaries, 'cases': rows, 'current_invocation_seconds': time.perf_counter() - start,
                  'scope': config['reference_policy']}
        write_json(output / 'results.json', report)
        print(len(rows), row['case_id'], {v: round(row['variants'][v]['result']['metrics']['cost'], 6) for v in row['variants']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.output)
    else: run(args.output)
