"""Resumable full-dev raw diagnostics and one direct steady evaluation per best."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from .io import HERE, ROOT, sha, save, snapshot, exclusive
from .data import load_rows
from .model import GeometryModel
from .inference import predict
from .freeze import frozen_checkpoint
from .diagnostics import geometry_checks, distribution

STEADY = dict(relaxed=False, mode=0, adaptive=True, fast_budget=False, late_extension=True,
              plateau=True, robust_seeds=True, certified=True, quick_project=True,
              quadratic_project=True, incumbent_project=True, snap=1e-4)


def stopped(folder):
    return (HERE/'runs/STOP').exists() or (folder/'STOP').exists()


def initialize(checkpoint_path):
    global worker_model, worker_rows, worker_initialization
    from .canonical_native import warmup
    from .train import seed_all
    started = time.perf_counter()
    seed_all(0, 2)
    warmup()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    worker_model = GeometryModel(**state['model_config']).eval()
    worker_model.load_state_dict(state['model'])
    worker_rows = load_rows('dev_monitor500')
    worker_initialization = time.perf_counter()-started


def evaluate_row(index, row, model):
    from .solver import optimize
    from .train import seed_all
    seed_all(index, 2)
    case = row['case']
    start = time.perf_counter()
    record = {'index': index, 'member': list(row['member']), 'n': case.block_count}
    try:
        draft, inference_seconds = predict(model, case.solver_inputs)
        positions, stats = optimize(draft, case.solver_inputs, **STEADY)
        elapsed = time.perf_counter()-start
        # Scoring and diagnostics occur outside the complete-solve timer.
        final_score = evaluate_case(case, positions.tolist(), elapsed)
        raw_score = evaluate_case(case, draft.tolist(), inference_seconds)
        _, hard, diagnostic = geometry_checks(draft, case.solver_inputs)
        local = np.sqrt(np.asarray(case.solver_inputs[1], dtype=float))
        center_move = np.linalg.norm((positions[:, :2]+positions[:, 2:]/2)-
                                    (draft[:, :2]+draft[:, 2:]/2), axis=1)/local
        trace = stats.get('trace', [])
        record.update(seconds=elapsed, metrics=final_score['metrics'], raw_metrics=raw_score['metrics'],
                      raw_hard_legal=hard['is_feasible'], draft_diagnostics=diagnostic,
                      positions=positions.tolist(), trace=trace, row_fallback=bool(stats['row_retained']),
                      rescue_used=any(any(s in str(r.get('phase', '')) for s in ('rescue', 'packed')) for r in trace),
                      inference_seconds=inference_seconds, postprocess_seconds=stats['seconds'],
                      mean_center_move_local=float(center_move.mean()), max_center_move_local=float(center_move.max()),
                      worker_pid=os.getpid())
    except Exception as error:
        record.update(seconds=time.perf_counter()-start, metrics={'cost': 10., 'is_feasible': False},
                      error=repr(error))
    return record


def evaluate(index):
    result = evaluate_row(index, worker_rows[index], worker_model)
    result['resident_initialization_seconds'] = worker_initialization
    return result


def summarize(records):
    q = np.asarray([r['metrics']['cost'] for r in records])
    n = np.asarray([r['n'] for r in records]); w = np.exp((n-n.max())/12)
    seconds = np.asarray([r['seconds'] for r in records])
    result = {'count': len(records), 'hard_legal': sum(r['metrics']['is_feasible'] for r in records),
              'mean_Q': float(q.mean()), 'weighted_Q': float(np.average(q, weights=w)),
              'p95_Q': float(np.percentile(q, 95)), 'max_Q': float(q.max()),
              'worst5pct_mean_Q': float(np.sort(q)[-max(1, int(np.ceil(.05*len(q)))):].mean()),
              'seconds_mean': float(seconds.mean()), 'seconds_p95': float(np.percentile(seconds, 95)),
              'row_fallbacks': sum(r.get('row_fallback', False) for r in records),
              'rescue_cases': sum(r.get('rescue_used', False) for r in records),
              'errors': sum('error' in r for r in records)}
    for key in ('hpwl_gap', 'area_gap', 'boundary_violations', 'grouping_violations', 'mib_violations'):
        returned = [r['metrics'][key] for r in records if key in r['metrics']]
        result[key+'_mean_returned'] = float(np.mean(returned)) if returned else None
    for key in ('inference_seconds', 'postprocess_seconds', 'mean_center_move_local', 'max_center_move_local'):
        returned = [r[key] for r in records if key in r]
        result[key+'_mean_returned'] = float(np.mean(returned)) if returned else None
    return result


def full_dev(output, state, identity):
    from .train import monitor_losses, seed_all
    target = output/'full_dev.json'; partial_path = output/'full_dev_partial.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))['checkpoint_sha256'] != identity['checkpoint_sha256']:
            raise ValueError('Full-dev checkpoint changed')
        return True
    device = state['identity']['device']
    seed_all(state['identity']['seed'], 4)
    model = GeometryModel(**state['model_config']).to(device).eval(); model.load_state_dict(state['model'])
    rows = load_rows('dev')
    chunks = json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else []
    cursor = sum(c['count'] for c in chunks)
    if cursor > len(rows):
        raise ValueError('Invalid full-dev cursor')
    for start in range(cursor, len(rows), 32):
        if stopped(output):
            return False
        value = monitor_losses(model, rows[start:start+32], device, state['identity']['arm'], 2., 32,
                               return_records=True)
        chunks.append(value); save(partial_path, chunks)
        if len(chunks)%10 == 0:
            print('Full-dev raw diagnostics:', min(start+32, len(rows)), '/', len(rows), flush=True)
    records = [r for chunk in chunks for r in chunk['diagnostic_records']]
    numeric = [k for k, v in chunks[0].items() if isinstance(v, (int, float)) and
               k not in ('count', 'raw_hard_legal', 'raw_hard_legal_rate')]
    values = {k: sum(c[k]*c['count'] for c in chunks)/len(rows) for k in numeric}
    values.update(count=len(rows), raw_hard_legal=sum(c['raw_hard_legal'] for c in chunks))
    values['raw_hard_legal_rate'] = values['raw_hard_legal']/len(rows)
    values['draft_diagnostics'] = distribution(records, [k for k in records[0] if k != 'member'])
    save(target, {'checkpoint_sha256': identity['checkpoint_sha256'], 'losses': values,
                  'count': len(rows), 'postprocessing_calls': 0, 'test_read': False})
    partial_path.unlink()
    return True


def terminal(output, identity):
    folder = output/'terminal'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'results.json').exists():
        return True
    path = folder/'partial.json'
    records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    metadata = ReferenceLayouts('dev_monitor500').records
    done = {r['index'] for r in records}
    if len(done) != len(records) or any(i not in range(500) for i in done):
        raise ValueError('Invalid evaluation cursor')
    if any(r['member'] != list(metadata[r['index']]['member']) for r in records):
        raise ValueError('Evaluation member changed')
    save(folder/'config.json', {**identity, 'steady': STEADY,
         'timing': 'public features + CPU inference + decode + all repair/rescue; excludes load, JIT and scoring'})
    snapshot(folder/'source_snapshot.zip', identity)
    previous_env = os.environ.get('E41_THREADS')
    os.environ['E41_THREADS'] = '2'
    try:
        todo = iter(i for i in range(500) if i not in done); pending = {}
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=initialize,
                                 initargs=(ROOT/identity['checkpoint'],)) as pool:
            def submit():
                if stopped(output):
                    return
                index = next(todo, None)
                if index is not None:
                    pending[pool.submit(evaluate, index)] = index
            for _ in range(2):
                submit()
            while pending:
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future); records.append(future.result()); save(path, records); submit()
                    if len(records)%25 == 0:
                        print('Terminal steady:', len(records), '/ 500', flush=True)
    finally:
        if previous_env is None:
            os.environ.pop('E41_THREADS', None)
        else:
            os.environ['E41_THREADS'] = previous_env
    if len(records) != 500:
        return False
    records.sort(key=lambda r: r['index'])
    summary = summarize(records)
    save(folder/'results.json', {'summary': summary, 'records': records, 'test_read': False})
    save(folder/'summary.json', {'checkpoint_sha256': identity['checkpoint_sha256'], **summary,
                               'test_read': False, 'neutral_runtime_development_only': True})
    path.unlink(); print(json.dumps(summary), flush=True)
    return True


def run(output, *, baseline=False, stage='all'):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE/'runs'):
        raise ValueError('Output must stay in E41/runs')
    with exclusive(output.with_suffix('.lock')):
        state, identity = frozen_checkpoint(output, baseline=baseline)
        if stopped(output):
            return
        if not baseline and stage in ('all', 'raw'):
            if not full_dev(output, state, identity):
                return
        if stage in ('all', 'terminal'):
            terminal(output, identity)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--stage', choices=('all', 'raw', 'terminal'), default='all')
    args = ap.parse_args()
    # Ctrl+C uses the same safe stop path as a STOP file; keep completed records.
    old = signal.signal(signal.SIGINT, lambda *_: (args.output/'STOP').touch())
    try:
        run(args.output, baseline=args.baseline, stage=args.stage)
    finally:
        signal.signal(signal.SIGINT, old)


if __name__ == '__main__':
    main()
