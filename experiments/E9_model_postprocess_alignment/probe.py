"""Five fixed dev cases: controlled draft changes and reference diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from datasets.paths import ROOT
from datasets.worksets import LayoutReader, workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from .records import save, append, read_rows, run_path, provenance, snapshot, sha

WORKERS = (0, 24, 49, 74, 99)
VARIANTS = ('raw', 'public_mib', 'oracle_shape', 'oracle_center', 'oracle_layout')
E8 = ROOT / 'experiments/E8_fast_robust_postprocess/runs'
DRAFTS = E8 / '20261018_stageA_dev100_seed0/drafts.jsonl'
BASELINE = E8 / '20261018_robust_dev100_seed0/cases.jsonl'


def build_case(raw):
    return _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6], raw.reference_positions, raw.stored_metrics)


def worker_call(output, inputs, draft, seconds=5.):
    begin = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='public_worker_', dir=output) as folder:
        request, response = Path(folder) / 'request.json', Path(folder) / 'response.json'
        save(request, {'inputs': [int(inputs[0]), *[np.asarray(v).tolist() for v in inputs[1:]]],
                       'draft': np.asarray(draft).tolist(), 'seconds': seconds})
        try:
            child = subprocess.run([sys.executable, '-B', '-m', 'experiments.E9_model_postprocess_alignment.worker', str(request), str(response)],
                                   cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=seconds + 10)
            if child.returncode or not response.exists():
                raise RuntimeError(child.stderr[-2500:] or f'worker exit {child.returncode}')
            result = json.loads(response.read_text(encoding='utf-8'))
        except Exception as exc:
            result = {'error': {'type': type(exc).__name__, 'message': str(exc)}, 'solve_seconds': time.perf_counter() - begin}
    result['worker_wall_seconds'] = time.perf_counter() - begin
    return result


def evaluate_output(case, result):
    if 'error' not in result:
        try:
            result['final'] = evaluate_case(case, result.pop('positions'), result['solve_seconds'])
            return result
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['final'] = {'metrics': {'is_feasible': False, 'cost': 10.}, 'positions': None}
    return result


def summaries(rows, variants):
    from scoring.official import official_evaluator
    output = {}
    for name in variants:
        values = [r['variants'][name] for r in rows if name in r['variants']]
        if not values:
            continue
        metrics = [v['final']['metrics'] for v in values]
        q = [m['cost'] for m in metrics]
        ids = [r for r in rows if name in r['variants']]
        output[name] = {'n': len(values), 'Q_mean': float(np.mean(q)), 'Q_max': max(q),
                        'Q_weighted': float(official_evaluator().compute_total_score(q, [r['block_count'] for r in ids])),
                        'feasible': sum(m['is_feasible'] for m in metrics),
                        'errors': sum('error' in v for v in values),
                        'postprocess_seconds_mean': float(np.mean([v['solve_seconds'] for v in values])),
                        **{key + '_mean': float(np.mean([m[key] for m in metrics if key in m])) if any(key in m for m in metrics) else None
                           for key in ['hpwl_gap', 'area_gap', 'boundary_violations', 'grouping_violations', 'mib_violations']}}
    return output


def mib_details(inputs, positions):
    c = np.asarray(inputs[5]); areas = np.asarray(inputs[1]); specs = np.asarray(inputs[6])
    locked = (c[:, 0] != 0) | (c[:, 1] != 0)
    details = []
    for group in sorted(set(c[:, 2].astype(int)) - {0}):
        ids = np.flatnonzero(c[:, 2] == group)
        details.append({'group': int(group), 'blocks': ids.tolist(), 'areas': areas[ids].tolist(),
                        'locked_blocks': ids[locked[ids]].tolist(),
                        'shared_area_interval': [float((areas[ids] * .99).max()), float((areas[ids] * 1.01).min())],
                        'shapes': {k: np.asarray(v)[ids, 2:].tolist() for k, v in positions.items()},
                        'distinct_shapes_round4': {k: len({tuple(round(float(a), 4) for a in x) for x in np.asarray(v)[ids, 2:]}) for k, v in positions.items()}})
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=900.)
    args = parser.parse_args()
    if not 0 < args.budget_seconds < 1200:
        parser.error('This five-case probe must stay below 20 minutes')
    output = run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = workset_manifest(); reader = LayoutReader(manifest, 'dev_quality')
    indices = [i for i, member in enumerate(reader.members) if int(member[0].split('/')[0].split('_')[1]) in WORKERS]
    assert len(indices) == 5
    config = {**provenance(), 'members': [reader.members[i] for i in indices], 'workers': WORKERS,
              'variants': VARIANTS, 'postprocessor': 'Unmodified E8 conic_robust dependency copy',
              'solver_seconds': 5., 'seed': 0, 'budget_seconds': args.budget_seconds,
              'workset_sha256': manifest['manifest_sha256'], 'test_payload_reads': 0,
              'draft_source': str(DRAFTS), 'draft_source_sha256': sha(DRAFTS),
              'baseline_source': str(BASELINE), 'baseline_source_sha256': sha(BASELINE),
              'oracle_disclosure': 'oracle_shape/center/layout use dev reference only for diagnostics; none is deployable. No training performed.'}
    save(output / 'config.json', config); snapshot(output, config['source_sha256'])
    drafts = {r['case_id']: r for r in read_rows(DRAFTS)}
    baselines = {r['case_id']: r for r in read_rows(BASELINE)}
    rows = []
    for j, index in enumerate(indices):
        if time.perf_counter() - start > args.budget_seconds - 30 or (output / 'STOP').exists():
            break
        raw_case = reader.case(index); case = build_case(raw_case)
        draft = np.array(drafts[case.case_id]['draft']); reference = np.array(case.reference_positions)
        from .backend.frozen.packing import prepare_problem, _dimensions
        problem = prepare_problem(*case.solver_inputs)
        w, h = _dimensions(problem, draft[:, 2] / draft[:, 3], True)
        public = draft.copy(); public[:, 2:] = np.column_stack((w, h))
        old_center = draft[:, :2] + draft[:, 2:] / 2
        shape = np.column_stack(((old_center - reference[:, 2:] / 2).clip(0), reference[:, 2:]))
        center = np.column_stack(((reference[:, :2] + reference[:, 2:] / 2 - draft[:, 2:] / 2).clip(0), draft[:, 2:]))
        for variant in (public, shape, center):
            variant[problem.fixed | problem.preplaced, 2:] = problem.specs[problem.fixed | problem.preplaced, 2:]
            variant[problem.preplaced] = problem.specs[problem.preplaced]
        proposals = dict(raw=draft, public_mib=public, oracle_shape=shape, oracle_center=center, oracle_layout=reference)
        row = {'case_id': case.case_id, 'source': case.source, 'block_count': case.block_count,
               'constraints': np.asarray(case.solver_inputs[5]).tolist(), 'reference_positions': reference.tolist(),
               'reference_metrics': evaluate_case(case, reference.tolist(), 0.)['metrics'],
               'variants': {}, 'cached_baseline_Q': baselines[case.case_id]['variants']['conic_robust']['final']['metrics']['cost']}
        # Rotate execution order to avoid always giving one variant the warm start.
        order = list(VARIANTS[j % len(VARIANTS):] + VARIANTS[:j % len(VARIANTS)])
        for name in order:
            if time.perf_counter() - start > args.budget_seconds - 20 or (output / 'STOP').exists():
                break
            result = worker_call(output, case.solver_inputs, proposals[name])
            row['variants'][name] = evaluate_output(case, result)
            print(json.dumps({'case': j, 'variant': name, 'Q': row['variants'][name]['final']['metrics']['cost'],
                              'error': row['variants'][name].get('error')}), flush=True)
        positions = {'reference': reference, 'raw': draft}
        positions.update({k: v['final']['positions'] for k, v in row['variants'].items() if v['final']['positions'] is not None})
        row['mib_details'] = mib_details(case.solver_inputs, positions)
        append(output / 'cases.jsonl', row); rows.append(row)
        save(output / 'results.json', {'status': 'running', 'test_payload_reads': 0, 'summary': summaries(rows, VARIANTS),
                                      'wall_seconds': time.perf_counter() - start})
    save(output / 'results.json', {'status': 'completed' if len(rows) == 5 and all(len(r['variants']) == len(VARIANTS) for r in rows) else 'stopped',
                                  'test_payload_reads': 0, 'summary': summaries(rows, VARIANTS),
                                  'reference_Q_mean': float(np.mean([r['reference_metrics']['cost'] for r in rows])) if rows else None,
                                  'wall_seconds': time.perf_counter() - start, 'scientific_scope': 'Five fixed development cases; mechanism diagnostics only.'})


if __name__ == '__main__':
    main()
