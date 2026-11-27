"""Compare saved, completed capacity arms against both fixed references; no inference."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS = ROOT
FOLDERS = {
    'base96x4': ROOT/'experiments/E38_model_architecture/runs/20261022_capacity_base96x4_seed0',
    'deep96x12': ROOT/'experiments/E38_model_architecture/runs/20261022_capacity_deep96x12_seed0',
    'rectangle': ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0',
    'full_tokens': ROOT/'experiments/E40_endpoint_constraint_tokens/runs/20261023_tokens_w96d12_r2_full_tokens_seed0',
    'rectangle_no_gap': ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_no_gap_w96d12_rectangle_no_gap_seed0',
}
ARMS = tuple(FOLDERS)
SOURCES = {}


def read(path):
    blob = path.read_bytes()
    SOURCES[path.relative_to(RUNS).as_posix()] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


def relative(candidate, control):
    return {key: candidate[key]/control[key]-1 for key in candidate
            if isinstance(candidate[key], (int, float)) and key in control
            and isinstance(control[key], (int, float)) and control[key] != 0}


def paired(candidate, control):
    by_member = {tuple(row['member']): row for row in control}
    assert len(by_member) == len(control) == len(candidate) == 500
    assert set(by_member) == {tuple(row['member']) for row in candidate}
    ordered = [by_member[tuple(row['member'])] for row in candidate]
    assert all(a['n'] == b['n'] and a['index'] == b['index'] for a, b in zip(candidate, ordered))
    delta = np.array([a['metrics']['cost']-b['metrics']['cost'] for a, b in zip(candidate, ordered)])
    seconds = np.array([a['seconds']-b['seconds'] for a, b in zip(candidate, ordered)])
    n = np.array([row['n'] for row in candidate])
    weights = np.exp((n-n.max())/12)
    rng = np.random.default_rng(3800)
    ids = rng.integers(0, len(delta), size=(2000, len(delta)))
    boot_q = (delta[ids]*weights[ids]).sum(axis=1)/weights[ids].sum(axis=1)
    contributions = delta*weights/weights.sum()
    top = np.argsort(np.abs(contributions))[-5:][::-1]
    return {'weighted_Q_delta': float(np.average(delta, weights=weights)),
            'case_bootstrap95_Q_delta': np.percentile(boot_q, [2.5, 97.5]).tolist(),
            'mean_seconds_delta': float(seconds.mean()),
            'case_bootstrap95_mean_seconds_delta': np.percentile(seconds[ids].mean(axis=1), [2.5, 97.5]).tolist(),
            'Q_better_equal_worse': [int((delta < -1e-9).sum()), int((abs(delta) <= 1e-9).sum()), int((delta > 1e-9).sum())],
            'largest_absolute_weighted_contributions': [
                {'index': candidate[i]['index'], 'member': candidate[i]['member'], 'n': int(n[i]),
                 'Q_delta': float(delta[i]), 'weighted_contribution': float(contributions[i])} for i in top],
            'uncertainty_limit': 'Paired case bootstrap measures case sampling only, not timed-solver run variability or training-seed uncertainty.'}


def main():
    completed = {}
    for arm in ARMS:
        folder = FOLDERS[arm]
        if not all((folder/name).exists() for name in ('frozen.json', 'full_dev.json', 'terminal/results.json')):
            continue
        terminal = read(folder/'terminal/results.json')
        if len(terminal['records']) != 500:
            continue
        result = read(folder/'results.json')
        full = read(folder/'full_dev.json')
        progress = read(folder/'progress.json')
        freeze = read(folder/'frozen.json')
        assert full['count'] == 2000
        best = next(row for row in progress['history'] if row['epoch'] == result['best_epoch'])
        completed[arm] = {'training': result, 'frozen': freeze, 'full_dev': full['losses'],
                          'best_monitor': best['dev_monitor'], 'best_train_monitor': best['train_monitor'],
                          'common_epoch42': next(({'epoch': r['epoch'], 'train_proxy': r['train_monitor']['selection_proxy'],
                              'dev_proxy': r['dev_monitor']['selection_proxy']} for r in progress['history'] if r['epoch'] == 42), None),
                          'terminal': terminal['summary'],
                          '_records': terminal['records']}
    base = completed['deep96x12']
    reference = read(ROOT/'experiments/E38_model_architecture/runs/20261022_capacity_e25_reference/terminal/results.json')
    for arm, row in completed.items():
        row['full_dev_relative_vs_control'] = relative(row['full_dev'], base['full_dev'])
        row['raw_draft_relative_vs_control'] = {
            key: relative(value, base['full_dev']['draft_diagnostics'][key])
            for key, value in row['full_dev']['draft_diagnostics'].items()}
        row['paired_vs_plain96x12'] = paired(row['_records'], base['_records'])
        row['paired_vs_initial_E25'] = paired(row['_records'], reference['records'])
        row['paired_vs_completed_arms'] = {name: paired(row['_records'], other['_records'])
                                           for name, other in completed.items() if name != arm}
    for row in completed.values():
        del row['_records']
    report = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'control_arm': 'deep96x12', 'completed_arms': completed,
              'source_sha256': SOURCES, 'test_read': False,
              'scope': 'Seed0 adaptive development; completed structure candidates and reused controls. No checkpoint selection from terminal Q.',
              'limitations': ['Read-only reuse of verified E38 96x12 control; structure candidates require their own diagnostics and ablations.',
                             'Case bootstrap excludes timed-solver variability documented in same_model_runtime_variation.json.',
                             'A lower raw proxy does not establish that a lightweight repair can resolve every draft.']}
    path = HERE/'structure_comparison.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps({arm: {'full_dev_proxy': row['full_dev']['selection_proxy'],
                           'full_dev_relative': row['full_dev_relative_vs_control']['selection_proxy'],
                           'paired_vs_control': row['paired_vs_plain96x12']}
                      for arm, row in completed.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
