"""Prespecified five-case budget/oracle diagnostic. Never reads local_test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import random
import subprocess
import sys
import threading
import time
import zipfile
import uuid

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
import torch
from datasets.paths import ROOT, resolved_data_paths
from datasets.worksets import LayoutReader, workset_manifest
from datasets.cases import _build_case
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .backend.frozen.packing import prepare_problem
from .backend.frozen.global_refine import topology, contact_tree

HERE = Path(__file__).resolve().parent
HISTORY = ROOT / 'history/20261018_E10_consolidation/experiments/E8_fast_robust_postprocess/runs'
DRAFTS = HISTORY / '20261018_stageA_dev100_seed0/drafts.jsonl'
TEACHERS = HISTORY / '20261018_robust_dev100_seed0/cases.jsonl'
WORKERS = (0, 24, 49, 74, 99)
VARIANTS = ('raw', 'reference_center', 'teacher_center', 'teacher_relations', 'teacher_full')
BUDGETS = (.1, .25, .4, 5.)
TOLERANCE = .002


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(content, encoding='utf-8')
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def save(path, value):
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_rows(path, rows):
    write_atomic(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


class Worker:
    def __init__(self, log, module='experiments.E11_short_budget_diagnosis.worker'):
        self.messages = queue.Queue()
        self.log = log.open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            [sys.executable, '-B', '-m', module], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf-8')
        def receive():
            for line in self.process.stdout:
                self.messages.put(line)
            self.messages.put(None)
        threading.Thread(target=receive, daemon=True).start()
        if not self.read(30).get('ready'):
            raise RuntimeError('Worker did not become ready')

    def read(self, timeout):
        line = self.messages.get(timeout=timeout)
        if line is None:
            raise RuntimeError('Worker exited; inspect worker.log')
        return json.loads(line)

    def call(self, payload):
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + '\n')
        self.process.stdin.flush()
        response = self.read(20)
        response['worker_wall_seconds'] = time.perf_counter() - start
        return response

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"quit":true}\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def center_swap(draft, target, inputs):
    out = draft.copy()
    out[:, :2] = np.maximum(0., target[:, :2] + target[:, 2:] / 2 - out[:, 2:] / 2)
    constraints, specs = np.asarray(inputs[5]), np.asarray(inputs[6])
    locked = constraints[:, :2].any(1)
    out[locked, 2:] = specs[locked, 2:]
    out[constraints[:, 1] != 0] = specs[constraints[:, 1] != 0]
    return out


def score(case, positions, seconds):
    result = evaluate_case(case, positions, seconds)
    result.pop('positions')
    return result


def record(case, response, charged_inference, layouts):
    if 'error' in response:
        return {**response, 'metrics': {'cost': 10., 'is_feasible': False}}
    stats = response.pop('stats')
    positions = response.pop('positions')
    key = hashlib.sha256(np.asarray(positions, dtype='<f8').tobytes()).hexdigest()[:20]
    layouts.setdefault(key, positions)
    result = score(case, positions, response['solve_seconds'])
    events = stats.pop('improvements', [])
    curve = []
    for event in events:
        event_result = score(case, event.pop('positions'), event['elapsed'])
        curve.append({**event, 'metrics': event_result['metrics']})
    traces = list(stats.get('trace', []))
    traces.extend(stats.get('rescue_refinement', {}).get('trace', []))
    stats.pop('rescue_refinement', None)
    native = sum(t.get('native_seconds', 0.) for t in traces)
    return {**response, **result, 'layout_key': key, 'stats': stats,
            'quality_curve_offline': curve,
            'conic_calls': sum(t.get('phase') == 'conic_shape' for t in traces),
            'native_conic_seconds': native,
            'charged_inference_seconds': charged_inference,
            'charged_total_seconds': charged_inference + response['worker_wall_seconds'],
            'ipc_and_serialization_seconds': response['worker_wall_seconds'] - response['solve_seconds']}


def summary(rows):
    output = {}
    for variant in VARIANTS:
        output[variant] = {}
        for budget in BUDGETS:
            entries = [(r, t) for r in rows for t in r['trials']
                       if t['variant'] == variant and t['budget_seconds'] == budget]
            if not entries:
                continue
            def mean(key):
                values = [t[key] for _, t in entries if key in t]
                return float(np.mean(values)) if values else None
            def metric(key):
                values = [t['metrics'][key] for _, t in entries if key in t['metrics']]
                return float(np.mean(values)) if values else None
            paired = [(r, t) for r, t in entries if 'baseline_q' in r]
            output[variant][str(budget)] = {
                'count': len(entries), 'Q_mean': metric('cost'),
                'Q_max': max(t['metrics']['cost'] for _, t in entries),
                'hard_legal_count': sum(t['metrics']['is_feasible'] for _, t in entries),
                'error_count': sum('error' in t for _, t in entries),
                'core_seconds_mean': mean('solve_seconds'),
                'worker_wall_seconds_mean': mean('worker_wall_seconds'),
                'charged_total_seconds_mean': mean('charged_total_seconds'),
                'charged_total_seconds_max': max(t.get('charged_total_seconds', 0.) for _, t in entries),
                'ipc_seconds_mean': mean('ipc_and_serialization_seconds'),
                'conic_calls_mean': mean('conic_calls'), 'native_conic_seconds_mean': mean('native_conic_seconds'),
                'budget_overruns': sum(t['solve_seconds'] > budget for _, t in entries),
                'paired_count': len(paired),
                'Q_delta_mean_vs_live_full_raw': float(np.mean([t['metrics']['cost'] - r['baseline_q'] for r, t in paired])) if paired else None,
                'strict_no_worse_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q'] + 1e-9 for r, t in paired),
                'within_002_count': sum(t['metrics']['is_feasible'] and t['metrics']['cost'] <= r['baseline_q'] + TOLERANCE for r, t in paired),
                'hpwl_gap_mean': metric('hpwl_gap'), 'area_gap_mean': metric('area_gap'),
                'boundary_mean': metric('boundary_violations'),
                'grouping_mean': metric('grouping_violations'), 'mib_mean': metric('mib_violations')}
    return output


def enrich(rows):
    for row in rows:
        baseline = next((t for t in row['trials'] if t['variant'] == 'raw' and t['budget_seconds'] == 5.), None)
        if baseline is None:
            continue
        row['baseline_q'] = baseline['metrics']['cost']
        for trial in row['trials']:
            trial['Q_delta_vs_live_full_raw'] = trial['metrics']['cost'] - row['baseline_q']
            for key, tolerance in [('strict_first_hit_seconds_offline', 1e-9), ('within_002_first_hit_seconds_offline', TOLERANCE)]:
                hits = [e['elapsed'] for e in trial.get('quality_curve_offline', [])
                        if e['metrics']['is_feasible'] and e['metrics']['cost'] <= row['baseline_q'] + tolerance]
                trial[key] = min(hits) if hits else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=900.)
    parser.add_argument('--resume', action='store_true', help='Explicitly complete missing cells of the same frozen diagnostic')
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 900:
        raise ValueError('This five-case diagnostic is capped at 900 seconds')
    previous = None
    prior_rows = []
    if args.resume:
        previous = json.loads((args.output/'results.json').read_text(encoding='utf-8'))
        if previous['status'] == 'completed':
            raise ValueError('Completed runs cannot be resumed')
        prior_rows = [json.loads(s) for s in (args.output/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(2)
    manifest = workset_manifest()
    reader = LayoutReader(manifest, 'dev_quality')
    indices = [i for i, (shard, _) in enumerate(reader.members)
               if int(shard.split('/')[0].split('_')[1]) in WORKERS]
    assert len(indices) == 5
    drafts = {r['case_id']: r for r in map(json.loads, DRAFTS.read_text(encoding='utf-8').splitlines())}
    teachers = {r['case_id']: r for r in map(json.loads, TEACHERS.read_text(encoding='utf-8').splitlines())}
    sources = sorted(HERE.rglob('*.py')) + [HERE/'README.md', HERE/'provenance.json']
    shared = [*sorted((ROOT/'datasets').glob('*.py')), *sorted((ROOT/'scoring').glob('*.py'))]
    import scipy
    import clarabel
    import highspy
    upstream = Path(official_evaluator().__file__)
    config = {'command': [sys.executable, *sys.argv], 'seed': 0, 'threads': 2,
              'started_local': time.strftime('%Y-%m-%d %H:%M:%S %z'),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
              'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
              'torch': torch.__version__, 'numpy': np.__version__, 'device': 'CPU',
              'scipy': scipy.__version__, 'clarabel': clarabel.__version__, 'highs': highspy.Highs().version(),
              'official_evaluator': {'path': str(upstream), 'sha256': sha(upstream)},
              'frozen_checkpoint_sha256': '47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28',
              'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources + shared},
              'split_sha256': sha(ROOT/'protocol/split_v1.json'), 'workset_sha256': sha(ROOT/'protocol/workset_v2.json'),
              'data_paths': resolved_data_paths(), 'budgets_seconds': BUDGETS, 'variants': VARIANTS,
              'workers': WORKERS, 'local_test_read': False, 'training': False,
              'absolute_quality_tolerance': TOLERANCE, 'wall_cap_seconds': args.budget_seconds,
              'cached_sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in (DRAFTS, TEACHERS)},
              'timing_scope': 'Steady-state core/IPC measured live; original inference reused and charged uniformly. Oracle construction, loading, scoring, plotting excluded. No deployable oracle model.',
              'reference_rule': 'Only reference_center uses free-block reference centers, explicitly diagnostic; teacher variants use cached E10 solution; raw worker receives masked public inputs only.'}
    invocation_path = args.output/'invocations.json'
    if args.resume:
        original = json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        for key in ('seed', 'threads', 'budgets_seconds', 'variants', 'workers', 'cached_sources', 'split_sha256', 'workset_sha256'):
            if json.dumps(config[key]) != json.dumps(original[key]):
                raise ValueError(f'Cannot change frozen diagnostic during resume: {key}')
        for path, digest in original['sources'].items():
            if '/backend/' in path and config['sources'].get(path) != digest:
                raise ValueError('Algorithm changes cannot be resumed into an existing run')
        invocations = json.loads(invocation_path.read_text(encoding='utf-8')) if invocation_path.exists() else [
            {'config': 'config.json', 'snapshot': 'source_snapshot.zip', 'status': previous['status'],
             'elapsed_seconds': previous['elapsed_seconds'], 'trial_count': previous['trial_count'],
             'error': original.get('fatal_error')}]
        suffix = f'_resume{len(invocations)}'
        config['resume_from_status'] = previous['status']
        config['prior_trial_count'] = previous['trial_count']
    else:
        invocations = []
        suffix = ''
    config_name = f'config{suffix}.json'
    snapshot_name = f'source_snapshot{suffix}.zip'
    save(args.output/config_name, config)
    with zipfile.ZipFile(args.output/snapshot_name, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources + shared:
            archive.write(path, path.relative_to(ROOT).as_posix())
    invocations.append({'config': config_name, 'snapshot': snapshot_name, 'status': 'running'})
    save(invocation_path, invocations)
    worker = Worker(args.output/f'worker{suffix}.log')
    rows = prior_rows; status = 'running'; rng = random.Random(0); warmed = False
    try:
        for index in indices:
            raw = reader.case(index)
            case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6], raw.reference_positions, raw.stored_metrics)
            cached = drafts[raw.case_id]
            teacher_record = teachers[raw.case_id]['variants']['conic_robust']['final']
            assert teacher_record['metrics']['is_feasible']
            draft = np.asarray(cached['draft']); teacher = np.asarray(teacher_record['positions'])
            inputs = case.solver_inputs
            locked = np.asarray(inputs[5])[:, :2].any(1)
            assert np.all(np.asarray(inputs[6])[~locked] == -1)
            problem = prepare_problem(*inputs)
            edges = topology(teacher, problem.constraints)
            contacts = contact_tree(teacher, problem, edges)
            guidance = {'edges': [[int(v) for v in e] for e in edges],
                        'contacts': [[int(v) for v in e] for e in contacts]}
            # A legal teacher supplies an acyclic, feasible full separation system.
            assert len(edges) == case.block_count * (case.block_count - 1) // 2
            assert all(teacher[i, a] + teacher[i, a+2] <= teacher[j, a] + 1e-5 for i, j, a in edges)
            proposals = {'raw': draft, 'reference_center': center_swap(draft, np.asarray(raw.reference_positions), inputs),
                         'teacher_center': center_swap(draft, teacher, inputs),
                         'teacher_relations': draft, 'teacher_full': teacher}
            serialized = [int(inputs[0]), *[np.asarray(v).tolist() for v in inputs[1:]]]
            row = {'case_id': raw.case_id, 'source': raw.source, 'block_count': case.block_count,
                   'constraints': np.asarray(inputs[5]).tolist(), 'layouts': {}, 'trials': [],
                   'cached_teacher_q': teacher_record['metrics']['cost'],
                   'cached_inference_seconds': cached['inference_seconds'],
                   'diagnostic_relation_counts': {'separations': len(edges), 'contacts': len(contacts)}}
            existing = next((r for r in rows if r['case_id'] == raw.case_id), None)
            if existing is None:
                rows.append(row)
            else:
                row = existing
            combinations = [(v, b) for v in VARIANTS for b in BUDGETS]
            rng.shuffle(combinations)
            finished = {(t['variant'], t['budget_seconds']) for t in row['trials']}
            combinations = [(v, b) for v, b in combinations if (v, b) not in finished]
            if not combinations:
                continue
            if not warmed:
                warmup = worker.call({'inputs': serialized, 'draft': draft.tolist(), 'seconds': .1})
                config['warmup_wall_seconds'] = warmup['worker_wall_seconds']
                save(args.output/config_name, config)
                warmed = True
            for variant, budget in combinations:
                if time.perf_counter() - begin + sum(i.get('elapsed_seconds', 0.) for i in invocations[:-1]) >= args.budget_seconds or (args.output/'STOP').exists():
                    status = 'budget_stop' if not (args.output/'STOP').exists() else 'user_stop'
                    break
                response = worker.call({'inputs': serialized, 'draft': proposals[variant].tolist(),
                                        'seconds': budget, 'guidance': guidance if variant == 'teacher_relations' else None})
                trial = record(case, response, cached['inference_seconds'], row['layouts'])
                row['trials'].append({'variant': variant, 'budget_seconds': budget, **trial})
                print(raw.case_id, variant, budget, f"Q={trial['metrics']['cost']:.6f}",
                      f"core={trial['solve_seconds']:.3f}s", flush=True)
                enrich(rows)
                save_rows(args.output/'cases.jsonl', rows)
                save(args.output/'results.json', {'status': 'running', 'elapsed_seconds': time.perf_counter()-begin,
                                                 'trial_count': sum(len(r['trials']) for r in rows), 'summary': summary(rows)})
            if status != 'running':
                break
        if status == 'running':
            status = 'completed'
    except Exception as exc:
        status = 'error'
        config['fatal_error'] = repr(exc)
        raise
    finally:
        worker.close()
        enrich(rows)
        save_rows(args.output/'cases.jsonl', rows)
        save(args.output/config_name, config)
        invocations[-1].update(status=status, elapsed_seconds=time.perf_counter()-begin,
                               trial_count=sum(len(r['trials']) for r in rows), error=config.get('fatal_error'))
        save(invocation_path, invocations)
        save(args.output/'results.json', {'status': status, 'elapsed_seconds': sum(i['elapsed_seconds'] for i in invocations),
                                         'case_count': len(rows), 'trial_count': sum(len(r['trials']) for r in rows),
                                         'summary': summary(rows),
                                         'scope': 'Five-case mechanism diagnostic; neutral-runtime Q; oracle is not a trained model; no generalization claim.'})


if __name__ == '__main__':
    main()
