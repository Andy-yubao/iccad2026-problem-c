"""Common real quality monitoring; no labels enter inference or repair."""
import time
import numpy as np
import torch
from scoring.evaluation import evaluate_case
from .data import collate
from .model import decode
from .backend.optimizer import optimize
from .backend.stable import polish


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, row, device):
    # Includes graph batching/transfer, network, decode, CPU transfer. Model load
    # and training-dataset file reads are excluded and reported elsewhere.
    synchronize(device)
    begin = time.perf_counter()
    graph, _ = collate([row], device)
    positions = (decode(model(graph), graph)[0] * graph['scale'][0]).cpu().double().numpy()
    inputs = row['case'].solver_inputs
    c, specs = np.asarray(inputs[5]), np.asarray(inputs[6])
    locked = c[:, :2].any(1)
    positions[locked, 2:] = specs[locked, 2:]
    positions[c[:, 1] != 0] = specs[c[:, 1] != 0]
    synchronize(device)
    return positions, time.perf_counter() - begin


def summary(records):
    costs = np.array([r['metrics']['cost'] for r in records])
    n = np.array([r['block_count'] for r in records])
    weights = np.exp(n / 12 - np.max(n / 12))
    return {'count': len(records), 'hard_legal': sum(r['metrics']['is_feasible'] for r in records),
            'mean_Q': float(costs.mean()), 'weighted_Q': float(np.average(costs, weights=weights)),
            'p95_Q': float(np.percentile(costs, 95)), 'max_Q': float(costs.max()),
            'Q_gt_1_8': int((costs > 1.8).sum()), 'Q_gt_2': int((costs > 2).sum()),
            'inference_seconds_mean': float(np.mean([r['inference_seconds'] for r in records])),
            'solve_seconds_mean': float(np.mean([r['solve_seconds'] for r in records])),
            'fallback_count': sum(r.get('fallback', False) for r in records),
            'error_count': sum('error' in r for r in records)}


def quality_monitor(model, rows, device, *, should_stop=lambda: False, existing=None, on_record=None):
    model.eval()
    records = list(existing or [])
    if len(records) > len(rows) or any(tuple(r['member']) != row['member'] for r, row in zip(records, rows)):
        raise ValueError('Partial quality evaluation panel mismatch')
    for row in rows[len(records):]:
        if should_stop():
            if on_record is not None:
                on_record(records, True)
            return None
        begin = time.perf_counter()
        inference = 0.
        try:
            draft, inference = predict(model, row, device)
            positions, stats = optimize(draft, row['case'].solver_inputs, seconds=5.)
            positions, stable = polish(draft, positions, row['case'].solver_inputs, stats, seconds=16.)
            solve = time.perf_counter() - begin
            # RAW and FINAL are scored separately; repaired legality does not
            # establish that the network itself learned a non-overlapping draft.
            raw = evaluate_case(row['case'], draft.tolist(), inference)['metrics']
            metrics = evaluate_case(row['case'], positions.tolist(), solve)['metrics']
            record = {'metrics': metrics, 'raw_metrics': raw,
                      'fallback': bool(stats.get('rescue_used') or stats.get('initial_row_retained')),
                      'stability_triggered': stable['triggered'], 'positions': positions.tolist(),
                      'inference_seconds': inference, 'solve_seconds': solve}
        except Exception as error:
            record = {'metrics': {'is_feasible': False, 'cost': 10.},
                      'error': repr(error), 'inference_seconds': inference,
                      'solve_seconds': time.perf_counter() - begin}
        record.update(member=list(row['member']), block_count=row['case'].block_count)
        records.append(record)
        if on_record is not None:
            on_record(records, False)
        if len(records) % 25 == 0:
            print(f'Quality evaluation: {len(records)}/{len(rows)}', flush=True)
    result = summary(records)
    result['raw_hard_legal'] = sum(r.get('raw_metrics', {}).get('is_feasible', False) for r in records)
    return {'summary': result, 'records': records}
