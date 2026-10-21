"""One frozen best: sequential online raw dev500 scoring, then full-dev proxy."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '4')
import numpy as np
import torch
from scoring.evaluation import evaluate_case
from .io import HERE, sha, save, snapshot, sources, exclusive
from .data import load_rows, collate, public_row, identity as cache_identity
from .model import inputs_only, load_trainable
from .train import REVISION, seed_all, make_model, physical_positions, monitor


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, row, device):
    synchronize(device); started = time.perf_counter()
    incoming = public_row(row['case'].solver_inputs, row['member'])
    graph = collate([incoming], device, online=True)
    trajectory = model(inputs_only(graph), return_trajectory=True)
    positions = physical_positions(trajectory[-1], graph, [row])[0]
    synchronize(device); seconds = time.perf_counter() - started
    # Additional baseline decoding is diagnostic work outside the candidate timer.
    baseline = physical_positions(trajectory[0], graph, [row])[0]
    return positions, baseline, seconds


def summary(records, key='metrics'):
    count = len(records)
    if not count:
        return {'count': 0}
    metrics = [r[key] for r in records]
    costs = np.asarray([m['cost'] for m in metrics]); n = np.asarray([r['block_count'] for r in records])
    weights = np.exp((n - n.max()) / 12); legal = np.asarray([m['is_feasible'] for m in metrics])
    result = {'count': count, 'hard_legal': int(legal.sum()), 'hard_legal_rate': float(legal.mean()),
              'weighted_hard_legal_rate': float(np.average(legal, weights=weights)),
              'mean_Q_all_cases': float(costs.mean()), 'weighted_Q_all_cases': float(np.average(costs, weights=weights)),
              'p95_Q': float(np.percentile(costs, 95)), 'max_Q': float(costs.max()),
              'conditional_legal_mean_Q': float(costs[legal].mean()) if legal.any() else None,
              'mean_solve_seconds': float(np.mean([r['solve_seconds'] for r in records])),
              'error_count': sum('error' in r for r in records), 'postprocessing_calls': 0}
    for name in ('hpwl_gap', 'area_gap', 'boundary_violations', 'grouping_violations', 'mib_violations', 'overlap_violations'):
        values = [m[name] for m in metrics if m.get(name) is not None]
        result['mean_' + name + '_returned'] = float(np.mean(values)) if values else None
    return result


def freeze_once(folder, frozen, resume=False):
    if folder.exists():
        if json.loads((folder / 'frozen.json').read_text(encoding='utf-8')) != frozen:
            raise ValueError('Cannot evaluate a second candidate in this training run')
        if not (folder / 'results.json').exists() and not resume:
            raise ValueError('Resume the same interrupted frozen evaluation')
    else:
        folder.mkdir(parents=True)
        save(folder / 'frozen.json', frozen)


def run(run_folder, device='cuda', resume=False):
    run_folder = Path(run_folder).resolve()
    if not run_folder.is_relative_to(HERE / 'runs'):
        raise ValueError('Use an E28 run')
    folder = run_folder / 'final_evaluation'
    with exclusive(run_folder.with_suffix('.lock')):
        seed_all(0)
        latest = torch.load(run_folder / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        ident = latest['identity']
        if not latest['finished'] or ident['revision'] != REVISION or ident['implementation_check']:
            raise ValueError('Only a formally finished model can enter final dev500')
        if sources() != ident['sources'] or cache_identity() != ident['cache_identity']:
            raise ValueError('Training source/data identity changed')
        path = run_folder / 'checkpoints/best.pt'
        best = torch.load(path, map_location='cpu', weights_only=False)
        if best['identity'] != ident:
            raise ValueError('Best/latest identities differ')
        digest = sha(path); rows = load_rows('dev_monitor500')
        if len(rows) != 500:
            raise ValueError('Final panel must have 500 members')
        frozen = {'checkpoint_sha256': digest, 'identity': ident, 'panel': 'dev_monitor500',
                  'members': [list(r['member']) for r in rows], 'device': device, 'cpu_threads': 4,
                  'postprocessing_calls': 0, 'selection': 'none; report all raw failures as Q=10'}
        freeze_once(folder, frozen, resume)
        if not (folder / 'config.json').exists():
            save(folder / 'config.json', {'command': [sys.executable, *sys.argv], 'frozen': frozen,
                 'timing': 'Sequential batch1; public feature construction, frozen base, refiner, decode and CPU transfer; loading and scoring excluded; baseline decoding diagnostic excluded'})
            snapshot(folder / 'source_snapshot.zip', ident)
        model = make_model(ident['arm'], device).eval(); load_trainable(model, best['model'])
        partial = folder / 'partial.json'
        existing = json.loads((folder / 'results.json').read_text(encoding='utf-8')) if (folder / 'results.json').exists() else None
        records = existing['records'] if existing else (json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else [])
        if len(records) > 500 or any(r['member'] != list(rows[i]['member']) for i, r in enumerate(records)):
            raise ValueError('Interrupted final panel differs')
        requested = [False]
        handler = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        stopped = lambda: requested[0] or (folder / 'STOP').exists() or (run_folder / 'STOP').exists() or (HERE / 'runs/STOP').exists()
        begin = time.perf_counter()
        def persist():
            save(partial, records)
            save(folder / 'progress.json', {'completed': len(records), 'total': 500,
                 'seconds_this_invocation': time.perf_counter() - begin, 'checkpoint_sha256': digest,
                 'test_read': False, 'postprocessing_calls': 0})
        try:
            for index in range(len(records), 500):
                if stopped():
                    persist(); return False
                row = rows[index]; started = time.perf_counter()
                try:
                    positions, initial, seconds = predict(model, row, device)
                    measured = evaluate_case(row['case'], positions, seconds)
                    base = evaluate_case(row['case'], initial, 0.)
                    record = {'member': list(row['member']), 'block_count': row['case'].block_count,
                              'metrics': measured['metrics'], 'baseline_metrics': base['metrics'],
                              'independent_hard_checks': measured['independent_hard_checks'],
                              'positions': positions.tolist(), 'solve_seconds': seconds,
                              'mean_center_movement': float(np.linalg.norm((positions[:, :2] + positions[:, 2:] / 2) -
                                                           (initial[:, :2] + initial[:, 2:] / 2), axis=1).mean())}
                except Exception as error:
                    failure = {'is_feasible': False, 'cost': 10.}
                    record = {'member': list(row['member']), 'block_count': row['case'].block_count,
                              'metrics': failure, 'baseline_metrics': failure, 'error': repr(error),
                              'solve_seconds': time.perf_counter() - started}
                records.append(record)
                if len(records) % 10 == 0:
                    persist(); print(f'raw final: {len(records)}/500', flush=True)
            if not existing:
                save(folder / 'results.json', {'status': 'complete', 'checkpoint_sha256': digest,
                     'summary': summary(records), 'frozen_base_summary': summary(records, 'baseline_metrics'),
                     'records': records, 'test_read': False, 'postprocessing_calls': 0,
                     'scope': 'MIB-derived adaptive development, neutral-time Q; raw output, not contest score'})
                if partial.exists():
                    partial.unlink()
            if not (run_folder / 'full_dev.json').exists() and not stopped():
                result = monitor(model, load_rows('dev'), device, should_stop=stopped)
                if result is None:
                    return False
                save(run_folder / 'full_dev.json', {'checkpoint_sha256': digest, 'metrics': result,
                     'test_read': False, 'postprocessing_calls': 0, 'used_for_selection': False})
            print(json.dumps(summary(records), ensure_ascii=False), flush=True)
            return (run_folder / 'full_dev.json').exists()
        finally:
            signal.signal(signal.SIGINT, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.run, args.device, args.resume)
