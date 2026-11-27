"""Finalize paired diagnostics from saved coordinates/metrics; no inference/Q."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.E90_auxiliary_weight_placer.io import HERE, save, sha, sources

RUN = HERE/'runs/20261031_aux3_seed0'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired(current, baseline, key, weighted_by_n=False):
    assert [r['member'] for r in current] == [r['member'] for r in baseline]
    a = np.array([r[key] for r in current], dtype=float)
    b = np.array([r[key] for r in baseline], dtype=float)
    w = np.array([r['n'] for r in current], dtype=float) if weighted_by_n else np.ones(len(a))
    groups = np.array([r['member'][0].split('/')[0] for r in current])
    labels = sorted(set(groups))
    numerator = np.array([((a-b)*w)[groups == g].sum() for g in labels])
    denominator = np.array([w[groups == g].sum() for g in labels])
    draw = np.random.default_rng(20261031).integers(0, len(labels), size=(5000, len(labels)))
    values = numerator[draw].sum(1)/denominator[draw].sum(1)
    return dict(count=len(a), current=float(np.average(a, weights=w)),
        baseline=float(np.average(b, weights=w)), difference=float(np.average(a-b, weights=w)),
        worker_bootstrap95=np.percentile(values, [2.5, 97.5]).tolist(),
        better=int(((a > b+1e-10) if key=='module_pass10_fraction' else (a < b-1e-10)).sum()),
        worse=int(((a < b-1e-10) if key=='module_pass10_fraction' else (a > b+1e-10)).sum()),
        equal=int((abs(a-b) <= 1e-10).sum()), weighted_by_module_count=weighted_by_n)


def main():
    frozen = read(RUN/'frozen.json')
    assert sources() == frozen['sources']
    assert sha(ROOT/frozen['checkpoint']) == frozen['checkpoint_sha256']
    assert sha(RUN/'checkpoints/latest.pt') == frozen['latest_checkpoint_sha256']
    raw = read(RUN/'full_dev.json')
    assert raw['checkpoint_sha256'] == frozen['checkpoint_sha256']
    q = read(RUN/'terminal/comparison.json')
    assert sha(RUN/'terminal/results.json') == q['current_sha256']
    terminal = read(RUN/'terminal/results.json')
    assert len(terminal['records']) == 500 and terminal['summary']['hard_legal'] == 500
    geometry = read(RUN/'inspection500.json')['geometry500']
    a = geometry['E90']['records']
    geometry_pairs = {name: {key: paired(a, geometry[name]['records'], key, weighted_by_n=(key == 'module_pass10_fraction'))
        for key in ('max_covered_fraction', 'module_pass10_fraction', 'pair_intersection_fraction',
                    'bbox_over_A', 'empty_fraction_of_bbox')}
        for name in ('E87', 'E74')}
    baseline_path = ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0/full_dev.json'
    old = read(baseline_path)
    current, baseline = raw['records'], old['records']
    assert len(current) == len(baseline) == raw['count'] == 2000
    assert [r['member'] for r in current] == [r['member'] for r in baseline]
    features_path = ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json'
    features = read(features_path)['records']
    assert [r['member'] for r in features] == [r['member'] for r in current]
    medians = {n: np.median([f['connection_density'] for f in features if f['n'] == n]) for n in {f['n'] for f in features}}
    ids = [i for i, f in enumerate(features) if f['pin_connected_fraction'] <= .3 and f['connection_density'] <= medians[f['n']]]
    assert len(ids) == 157
    raw_pairs = dict(all2000=paired(current, baseline, 'overlap_area_fraction'),
        weak_port_sparse157=paired([current[i] for i in ids], [baseline[i] for i in ids], 'overlap_area_fraction'))
    total_modules = sum(len(r['covered_fractions']) for r in current)
    pass_modules = total_modules-sum(r['N10'] for r in current)
    coverage = dict(module_count=total_modules, module_pass10_pooled=pass_modules/total_modules,
        layout_all_modules_pass10=sum(r['all_modules_pass10'] for r in current),
        worst_module_coverage_mean=float(np.mean([r['max_covered_fraction'] for r in current])),
        worst_module_coverage_p95=float(np.percentile([r['max_covered_fraction'] for r in current], 95)),
        worst_module_coverage_max=float(max(r['max_covered_fraction'] for r in current)))
    old_flash = read(ROOT/q['comparisons']['E87']['source'])
    by_id = {r['index']: r for r in old_flash['records']}
    counterexamples = []
    for r in terminal['records']:
        prior = by_id[r['index']]
        counterexamples.append(dict(index=r['index'], member=r['member'],
            E90_Q=r['metrics']['cost'], E87_Q=prior['metrics']['cost'],
            difference=r['metrics']['cost']-prior['metrics']['cost']))
    result = dict(status='complete_with_report_boundary_fix', best_epoch=12,
        checkpoint_sha256=frozen['checkpoint_sha256'], terminal_Q500=q['summary'],
        paired_Q500=q['comparisons'], draft_geometry500={k: {x: y for x, y in v.items() if x != 'records'} for k, v in geometry.items()},
        paired_draft_geometry500=geometry_pairs, paired_overlap2000_vs_E74=raw_pairs,
        E90_exact_coverage2000=coverage,
        top_Q_regressions_vs_E87=sorted(counterexamples, key=lambda r: r['difference'], reverse=True)[:5],
        full_dev_reporting_recovery=read(RUN/'raw_reporting_recovery.json'),
        sources_unchanged=True, training_checkpoints_unchanged=True, terminal_Q_unchanged=True,
        new_postprocessing_calls=0, new_model_inference_calls=0, test_read=False,
        limitations='Single-seed adaptive dev evidence; differing outer concurrency can affect timed solver search. '
                    'Same per-solve threads, algorithm, cases and scoring; not a formal contest result. '
                    'Original E87 full dev2000 was not evaluated; compare all three only on common500.')
    save(RUN/'inspection.json', result)
    print(json.dumps(dict(geometry_pairs=geometry_pairs, raw_pairs=raw_pairs, coverage2000=coverage,
                         worst_Q_regressions=result['top_Q_regressions_vs_E87'][:2]), indent=2))


if __name__ == '__main__':
    main()
