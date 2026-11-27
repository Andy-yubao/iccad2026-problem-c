"""Compare the completed fixed-panel assessment with saved baselines only."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.E90_auxiliary_weight_placer.io import save, sha

FOLDER = ROOT/'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/terminal'


def main():
    current_path = FOLDER/'results.json'
    current = json.loads(current_path.read_text(encoding='utf-8'))
    rows = sorted(current['records'], key=lambda r: r['index'])
    assert len(rows) == 500 and [r['index'] for r in rows] == list(range(500))
    n = np.array([r['n'] for r in rows])
    weight = np.exp((n-n.max())/12)
    q = np.array([r['metrics']['cost'] for r in rows])
    workers = np.array([r['member'][0].split('/')[0] for r in rows])
    unique = sorted(set(workers))
    comparisons = {}
    for name, run in (
        ('E87', 'E87_simple_supervised_placer/runs/20261030_simple_seed0'),
        ('E74', 'E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0'),
        ('E42', 'E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0')):
        path = ROOT/'experiments'/run/'terminal/results.json'
        baseline = json.loads(path.read_text(encoding='utf-8'))
        old = sorted(baseline['records'], key=lambda r: r['index'])
        assert len(old) == 500
        assert all(a['member'] == b['member'] and a['n'] == b['n'] for a, b in zip(rows, old))
        diff = q-np.array([r['metrics']['cost'] for r in old])
        sums = np.array([(weight[workers == g]*diff[workers == g]).sum() for g in unique])
        denominators = np.array([weight[workers == g].sum() for g in unique])
        rng = np.random.default_rng(20261031)
        samples = rng.integers(0, len(unique), size=(5000, len(unique)))
        estimates = sums[samples].sum(1)/denominators[samples].sum(1)
        comparisons[name] = dict(source=path.relative_to(ROOT).as_posix(), source_sha256=sha(path),
            summary=baseline['summary'], weighted_Q_difference=float(np.average(diff, weights=weight)),
            worker_bootstrap95=np.quantile(estimates, [.025, .975]).tolist(),
            worker_count=len(unique), bootstrap_repeats=5000, seed=20261031,
            Q_better=int((diff < -1e-12).sum()), Q_worse=int((diff > 1e-12).sum()),
            Q_tied=int((abs(diff) <= 1e-12).sum()))
    result = dict(current_source=current_path.relative_to(ROOT).as_posix(),
        current_sha256=sha(current_path), summary=current['summary'], comparisons=comparisons,
        lower_Q_is_better=True, test_read=False,
        interpretation='Repeatedly exposed development panel; neutral-runtime quality only. '
                       'One seed; both auxiliaries increased together, no separate attribution. '
                       'Current4x2, E87 1x2 and E74/E42 2x2 concurrency differ; same per-solve threads and solver, wall times not directly comparable; timed search may be affected by contention.')
    save(FOLDER/'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
