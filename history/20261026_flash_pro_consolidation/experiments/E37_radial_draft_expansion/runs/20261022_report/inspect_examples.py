"""Inspect saved E37 results and draw examples; no inference or optimization."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.E37_radial_draft_expansion.resources import configure
configure()
import json
import sqlite3
import zlib
import zipfile
import hashlib
import numpy as np
from experiments.E37_radial_draft_expansion.io import save, sha
from experiments.E37_radial_draft_expansion.cases import public_cases, WEIGHTS
from visualization.layouts import plot_case, geometry_details

OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]
FOLDER = OUT.parent / '20261022_paired100'


def main():
    con = sqlite3.connect('file:' + (FOLDER / 'outcomes.sqlite3').as_posix() + '?mode=ro', uri=True)
    records = [json.loads(zlib.decompress(row[0])) for row in con.execute('SELECT payload FROM records')]
    con.close()
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    arms = list(summary['summaries'])
    byarm = {arm: {r['index']: r for r in records if r['arm'] == arm} for arm in arms}
    cases, metadata, _ = public_cases('monitor')
    example_specs = []
    for arm in arms[1:]:
        effects = summary['paired_vs_reference'][arm]
        example_specs.append((arm, effects['worst_cases'][0]['index'], 'regression'))
    arm = 'radial5_rebase_restore'
    example_specs.append((arm, summary['paired_vs_reference'][arm]['best_cases'][0]['index'], 'improvement'))
    figures = []
    for arm, i, kind in example_specs:
        original = byarm['reference'][i]
        variant = byarm[arm][i]
        constraints = np.asarray(cases[i].solver_inputs[5])
        variants = {}
        for tag, r in [('reference', original), ('variant', variant)]:
            variants[tag] = {'positions': r['positions'], 'metrics': r['metrics'],
                             'soft_details': geometry_details(r['positions'], constraints)}
        name = f'case_{i}_{arm}_{kind}.png'
        plot_case({'panel': 'E37 paired dev100', 'case_id': str(metadata[i]['member']),
                   'constraints': constraints.tolist(), 'variants': variants,
                   'caption': 'Identical frozen draft and constraints; Q is a neutral-runtime development metric.'},
                  OUT / name, stages=(('reference', f'Original input ({original["seconds"]:.3f}s)'),
                                      ('variant', f'{arm} ({variant["seconds"]:.3f}s)')))
        figures.append({'file': name, 'index': i, 'arm': arm, 'kind': kind, 'n': original['n'],
                        'reference_Q': original['metrics']['cost'], 'variant_Q': variant['metrics']['cost'],
                        'reference_seconds': original['seconds'], 'variant_seconds': variant['seconds']})

    components = {}
    metrics = ('hpwl_gap', 'area_gap', 'boundary_violations', 'grouping_violations', 'mib_violations')
    for arm, rr in byarm.items():
        n = np.array([r['n'] for r in rr.values()])
        weights = np.exp((n - n.max()) / 12)
        components[arm] = {}
        for metric in metrics:
            values = [r['metrics'][metric] for r in rr.values()]
            components[arm][metric] = {'mean': float(np.mean(values)), 'weighted_mean': float(np.average(values, weights=weights))}
    strata = {}
    for arm in arms[1:]:
        strata[arm] = []
        for lo, hi in ((0, 60), (60, 90), (90, 110), (110, 10000)):
            ids = [i for i, r in byarm['reference'].items() if lo <= r['n'] < hi]
            dq = [byarm[arm][i]['metrics']['cost'] - byarm['reference'][i]['metrics']['cost'] for i in ids]
            strata[arm].append({'n_range': [lo, hi], 'count': len(ids), 'mean_delta_Q': float(np.mean(dq)) if ids else None})

    config = json.loads((FOLDER / 'config.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(FOLDER / 'source_snapshot.zip') as archive:
        assert set(archive.namelist()) == set(config['sources'])
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in config['sources'].items())
    assert all(sha(ROOT / name) == value for name, value in config['sources'].items())
    assert sha(WEIGHTS) == config['weights_sha256']
    original_drafts = {}
    for r in records:
        assert list(metadata[r['index']]['member']) == r['member']
        original_drafts.setdefault(r['index'], np.asarray(byarm['reference'][r['index']]['input_positions']))
        draft = original_drafts[r['index']]
        raw = np.asarray(r['raw_expanded_positions'])
        actual = np.asarray(r['input_positions'])
        assert np.array_equal(raw[:, 2:], draft[:, 2:]) and np.array_equal(actual[:, 2:], draft[:, 2:])
        if r['arm'] != 'reference':
            center = np.asarray(r['preprocessing']['origin'])
            assert np.allclose(raw[:, :2] + raw[:, 2:] / 2 - center,
                               1.05 * (draft[:, :2] + draft[:, 2:] / 2 - center), rtol=1e-11, atol=1e-10)
    baseline_results = ROOT / 'experiments/E25_integrated_baseline/runs/20261020_e10_geometry_v5_seed0/final_dev500/results.json'
    # Locate the original evidence by its recorded hash without touching any dataset/checkpoint.
    matches = [p for p in (ROOT / 'experiments/E25_integrated_baseline/runs').rglob('results.json')
               if sha(p) == '335b0dc1c85714608c41f8a703b78d6857dda60a2ceaab8b23eddcaeeaadd42d']
    assert len(matches) == 1
    data = {'figures': figures, 'final_components': components, 'descriptive_module_strata': strata,
            'checks': {'records': len(records), 'members': len(original_drafts), 'hard_legal': len(records),
                       'source_snapshot_and_live_sources_match': True, 'dimensions_unchanged': True,
                       'radial_factor_exact': True, 'weights_unchanged': True,
                       'original_E25_results_unchanged': matches[0].relative_to(ROOT).as_posix()},
            'source_database_sha256': sha(FOLDER / 'outcomes.sqlite3'), 'script_sha256': sha(Path(__file__)),
            'no_new_solves': True, 'test_read': False}
    save(OUT / 'examples.json', data)
    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    main()
