"""Check saved bound certificates, existing outputs and rounding counterexamples."""
import argparse
import math
from pathlib import Path
import numpy as np
from datasets.paths import ROOT
from .quality_bound import case_bound, SOURCE
from .records import save, read_rows, run_path, provenance, snapshot, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bound', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = run_path(args.output); output.mkdir(parents=True, exist_ok=False)
    import json
    bound = json.loads((args.bound / 'results.json').read_text(encoding='utf-8'))
    probe = read_rows(args.probe / 'cases.jsonl')
    cached = {r['case_id']: r for r in read_rows(SOURCE)}
    by_id = {r['case_id']: r for r in bound['cases']}
    checked_groups = checked_intervals = 0
    min_clearance = math.inf
    for row in bound['cases']:
        saved = cached[row['case_id']]
        labels = sorted(set(int(c[2]) for c in saved['public_constraints']) - {0})
        # The archived evaluator counts empty MIB IDs as -1; this proof applies
        # only after verifying that no such gaps occur in the fixed inputs.
        assert labels == list(range(1, max(labels, default=0) + 1))
        assert row['width_upper'] is not None and row['height_upper'] is not None
        assert row['Q_lower_bound'] <= row['E8_Q'] + 1e-10
        for group in row['groups']:
            intervals = group['round4_relaxed_intervals']; points = group['stabbing_points']
            # A feasible cover gives an upper bound on the relaxed minimum.
            assert all(any(a <= point <= b for point in points) for a, b in intervals)
            # Independent disjoint-interval packing gives a matching lower bound.
            witnesses = []; end = -math.inf
            for a, b in sorted(intervals, key=lambda t: t[1]):
                if a > end:
                    if witnesses:
                        min_clearance = min(min_clearance, a - end)
                    witnesses.append((a, b)); end = b
            assert len(witnesses) == len(points) == group['round4_min_mib_violations'] + 1
            assert all(witnesses[i][0] > witnesses[i - 1][1] for i in range(1, len(witnesses)))
            checked_groups += 1; checked_intervals += len(intervals)
    oracle_count = 0
    for row in probe:
        lower = by_id[row['case_id']]['Q_lower_bound']
        saved = cached[row['case_id']]
        assert row['constraints'] == saved['public_constraints']
        for group in row['mib_details']:
            assert group['areas'] == [saved['area_targets'][i] for i in group['blocks']]
        for variant in row['variants'].values():
            assert variant['final']['metrics']['cost'] + 1e-10 >= lower
            oracle_count += 1
    # Unequal areas can share a round4 shape through extreme aspect ratios.
    # This is a counterexample to a naive unconditional exact-area lower bound.
    wh = np.array([[.00001, 10000000.], [.00002, 10000000.]])
    assert len({tuple(round(float(v), 4) for v in shape) for shape in wh}) == 1
    assert np.allclose(wh.prod(1), [100., 200.])
    constraints = np.zeros((2, 5)); constraints[:, 2] = 1
    unanchored = case_bound([100, 200], constraints, np.full((2, 4), -1.), 300.)
    assert unanchored['Q_lower_bound'] == 1 and unanchored['width_upper'] is None
    config = {**provenance(), 'bound': str(args.bound), 'probe': str(args.probe),
              'bound_results_sha256': sha(args.bound / 'results.json'),
              'probe_cases_sha256': sha(args.probe / 'cases.jsonl'),
              'test_payload_reads': 0, 'dataset_tensor_reads': 0}
    save(output / 'config.json', config); snapshot(output, config['source_sha256'])
    inherited = json.loads((Path(__file__).parent / 'inherited_sources.json').read_text(encoding='utf-8'))
    unmodified=0
    for item in inherited['files']:
        assert sha(ROOT / item['source']) == item['sha256']
        if not item.get('modified'):
            assert sha(ROOT / item['destination']) == item['sha256'];unmodified+=1
        elif 'destination_sha256' in item:
            assert sha(ROOT / item['destination']) == item['destination_sha256']
    results = {'status': 'passed', 'bound_cases': len(bound['cases']), 'groups': checked_groups,
               'intervals': checked_intervals, 'minimum_disjoint_certificate_gap': min_clearance,
               'probe_outputs_not_contradicting_bound': oracle_count,
               'five_probe_public_constraints_and_mib_areas_match_saved100': True,
               'round4_extreme_aspect_counterexample_handled': True,
               'unmodified_copied_source_files': unmodified,
               'official_denominators_checked_in_bound_run': True,
               'dataset_tensor_reads': 0, 'test_payload_reads': 0}
    save(output / 'results.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
