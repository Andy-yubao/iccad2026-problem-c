"""Read saved development layouts/metrics only; no inference, solver, or scoring."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/20261023_draft_structure_audit'
AUDIT = ROOT / 'experiments/E39_rectangular_attention/runs/20261023_failure_mode_audit'
GALLERY = ROOT / 'experiments/E39_rectangular_attention/runs/20261023_rectangle_triptych100/drafts_local.json'
E39 = ROOT / 'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0/terminal/results.json'
E42 = ROOT / 'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json'
E35 = ROOT / 'experiments/E35_optimizer_attribution/runs/20261022_trajectory_features'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def describe(values):
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return None
    return {'mean': float(values.mean()), **dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
        np.percentile(values, [0, 25, 50, 75, 95, 100]).tolist()))}


def fixed_n_rho(rows, x, y):
    rows = [r for r in rows if r.get(x) is not None and r.get(y) is not None]
    a, b = rankdata([r[x] for r in rows]), rankdata([r[y] for r in rows])
    n = np.asarray([r['n'] for r in rows])
    for value in np.unique(n):
        ids = n == value
        a[ids] -= a[ids].mean(); b[ids] -= b[ids].mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else None


def geometry_retention(raw, final):
    """Descriptive geometry measurements, not extraction of a solver topology."""
    raw, final = np.asarray(raw, float), np.asarray(final, float)
    if raw.shape != final.shape or raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError('Invalid saved geometry')
    if not np.isfinite(raw).all() or not np.isfinite(final).all() or min(raw[:, 2:].min(), final[:, 2:].min()) <= 0:
        raise ValueError('Invalid saved geometry values')
    n = len(raw)
    units = np.sqrt(raw[:, 2:].prod(1))
    i, j = np.triu_indices(n, 1)
    pair_unit = (units[i] + units[j]) / 2
    def gaps(z):
        return np.column_stack((z[j, 0] - z[i, 0] - z[i, 2],
                                z[i, 0] - z[j, 0] - z[j, 2],
                                z[j, 1] - z[i, 1] - z[i, 3],
                                z[i, 1] - z[j, 1] - z[j, 3]))
    before, after = gaps(raw), gaps(final)
    valid = before >= -1e-6
    originally_separated = valid.any(1)
    kept = (valid & (after >= -1e-6)).any(1)
    robust = before >= .05 * pair_unit[:, None]
    robust_pairs = robust.any(1)
    robust_kept = (robust & (after >= -1e-6)).any(1)
    c0, c1 = raw[:, :2] + raw[:, 2:] / 2, final[:, :2] + final[:, 2:] / 2
    d0, d1 = c0[j] - c0[i], c1[j] - c1[i]
    rank_mask = np.abs(d0) > .05 * pair_unit[:, None]
    rank_kept = np.sign(d0) == np.sign(d1)
    def nearest(c):
        distance = ((c[:, None] - c[None, :]) ** 2).sum(2)
        np.fill_diagonal(distance, np.inf)
        return np.argsort(distance, axis=1, kind='stable')[:, :min(5, n - 1)]
    first, second = nearest(c0), nearest(c1)
    neighbors = np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(first, second)])
    displacement = c1 - c0
    mean = np.linalg.norm(displacement, axis=1) / units
    # Median is a robust translation diagnostic; it grants no motion to anchors.
    centered = np.linalg.norm(displacement - np.median(displacement, axis=0), axis=1) / units
    return {'originally_separated_pair_fraction': float(originally_separated.mean()),
            'originally_separated_pair_count': int(originally_separated.sum()),
            'separation_retention': float(kept[originally_separated].mean()) if originally_separated.any() else None,
            'robust_separated_pair_count': int(robust_pairs.sum()),
            'robust_separation_retention': float(robust_kept[robust_pairs].mean()) if robust_pairs.any() else None,
            'center_order_retention': float(rank_kept[rank_mask].mean()) if rank_mask.any() else None,
            'geometric_neighbor5_retention': float(neighbors),
            'center_mean_raw_area_units': float(mean.mean()),
            'center_detranslated_raw_area_units': float(centered.mean())}


def cohort(rows):
    keys = ['center_move', 'log_ratio_change', 'separation_retention', 'robust_separation_retention',
            'center_order_retention', 'geometric_neighbor5_retention', 'raw_overlap',
            'preplaced_overlap_area_fraction', 'failed_shape_calls', 'final_Q']
    return {'count': len(rows), 'mean_n': float(np.mean([r['n'] for r in rows])) if rows else None,
            **{k: describe([r[k] for r in rows]) for k in keys if rows and k in rows[0]}}


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    sources = [GALLERY, E39, E42, AUDIT / 'drafts_additional_local.npz',
               AUDIT / 'case_features_enriched.csv', E35 / 'cases.json', E35 / 'steps.json']
    parity_files = ['solver.py', 'fast_geometry.py', 'projection.py', 'backend/rescue.py',
                    'backend/frozen/feasible.py', 'backend/frozen/packing.py']
    sources += [ROOT / 'experiments' / family / name
                for family in ('E25_integrated_baseline', 'E42_joint_regularization') for name in parity_files]
    sources += [ROOT / 'experiments/E25_integrated_baseline/run.py',
                ROOT / 'experiments/E42_joint_regularization/final.py']
    before_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    with (AUDIT / 'case_features_enriched.csv').open(encoding='utf-8-sig') as handle:
        table = list(csv.DictReader(handle))
    for row in table:
        for key, value in list(row.items()):
            if key not in ('member', 'shard', 'worker'):
                row[key] = None if value == '' else float(value)
        row['index'], row['n'] = int(row['index']), int(row['n'])
    raw = {r['index']: np.asarray(r['raw_positions'], float) for r in read(GALLERY)}
    with np.load(AUDIT / 'drafts_additional_local.npz') as extra:
        raw.update({int(key): extra[key] for key in extra.files})
    final39 = {r['index']: r for r in read(E39)['records']}
    final42 = {r['index']: r for r in read(E42)['records']}
    if len(table) != 500 or set(raw) != set(final39) or set(final39) != set(final42):
        raise ValueError('Expected exactly the same existing 500 development cases')
    scalar42 = []
    for row in table:
        index = row['index']; a, b = final39[index], final42[index]
        if a['member'] != b['member'] or a['n'] != b['n'] or a['n'] != row['n']:
            raise ValueError('Model comparison membership mismatch')
        # The CSV uses shard:row; verify it before borrowing public metadata.
        shard, sample_index = row['member'].rsplit(':', 1)
        if [shard, int(sample_index)] != a['member']:
            raise ValueError('CSV member differs from saved terminal evaluation')
        row.update(geometry_retention(raw[index], a['positions']))
        if abs(row['center_move'] - a['mean_center_move_local']) > 1e-12:
            raise AssertionError('Source CSV and terminal displacement differ')
        # Public target areas and immutable hard WH need not be exactly equal
        # after the registered MIB transform. Keep the two units explicit.
        row['raw_area_vs_input_area_move_difference'] = row['center_mean_raw_area_units'] - row['center_move']
        row['separation_loss'] = 1 - row['separation_retention']
        row['neighbor_loss'] = 1 - row['geometric_neighbor5_retention']
        phases = b.get('trace', [])
        shape = [v for v in phases if v['phase'] == 'shape']
        scalar42.append({'index': index, 'member': b['member'], 'n': b['n'],
            **{key: row[key] for key in ('preplaced_fraction', 'group_fraction', 'mib_fraction', 'boundary_fraction')},
            'raw_overlap': b['draft_diagnostics']['overlap_area_fraction'],
            'center_move': b['mean_center_move_local'], 'max_move': b['max_center_move_local'],
            'final_Q': b['metrics']['cost'], 'rescue': int(b['rescue_used']),
            'early_projection': int(any(v['phase'] == 'early_projection' for v in phases)),
            'failed_shape_calls': sum(not v['legal'] for v in shape),
            'postprocess_seconds': b['postprocess_seconds']})
    groups = {'all': cohort(table),
              'ordinary': cohort([r for r in table if not r['early_projection']]),
              'early_projection': cohort([r for r in table if r['early_projection']]),
              'packing_rescue': cohort([r for r in table if r['rescue']])}
    associations = [{'x': x, 'y': y, 'within_n_rank_rho': fixed_n_rho(table, x, y)}
        for x in ('raw_overlap', 'preplaced_overlap_area_fraction', 'preplaced_fraction',
                  'group_fraction', 'mib_init_log_ratio_change')
        for y in ('center_move', 'separation_loss', 'neighbor_loss', 'failed_shape_calls', 'final_Q')]
    cases35, steps35 = read(E35 / 'cases.json'), read(E35 / 'steps.json')
    trajectories = {
        'count': len(cases35),
        'first_shape_move': describe([r['root_first_move']['center_mean'] for r in cases35]),
        'final_move': describe([r['draft_to_final']['center_mean'] for r in cases35]),
        'final_detranslated_move': describe([r['draft_to_final']['center_detranslated_mean'] for r in cases35]),
        'topology_flip_per_step': describe([r['topology_flip_fraction'] for r in steps35]),
        'contact_tree_replacement_per_step': describe([r['contact_tree_replacement_fraction'] for r in steps35]),
        'final_ratio_changed_2x_fraction': describe([r['draft_to_final']['AR_factor_ge_2_fraction'] for r in cases35]),
        'first_legal_step': describe([r['first_legal_shape_step'] for r in cases35]),
        'rescues': [{key: r[key] for key in ('index', 'n', 'root_shape_calls', 'root_legal_shapes',
                       'root_topology_flip_fraction_mean', 'root_contact_tree_replacement_mean',
                       'root_shape_cumulative_center', 'root_last_shape_net', 'metrics')}
                    for r in cases35 if r['rescue']]}
    parity = {name: sha(ROOT / 'experiments/E25_integrated_baseline' / name) ==
                    sha(ROOT / 'experiments/E42_joint_regularization' / name) for name in parity_files}
    control42 = {'count': len(scalar42), 'center_move': describe([r['center_move'] for r in scalar42]),
                 'rescue_count': sum(r['rescue'] for r in scalar42),
                 'ordinary_count': sum(not r['early_projection'] for r in scalar42),
                 'within_n_associations': [{'x': x, 'y': y, 'within_n_rank_rho': fixed_n_rho(scalar42, x, y)}
                    for x in ('raw_overlap', 'preplaced_fraction', 'group_fraction')
                    for y in ('center_move', 'failed_shape_calls', 'final_Q')]}
    examples = [next(r for r in table if r['index'] == i) for i in (98, 368, 443, 164, 316)]
    examples += [min(table, key=lambda r: r['center_move'])]
    if before_hashes != {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}:
        raise AssertionError('Historical evidence changed during read-only audit')
    summary = {'scope': 'saved adaptive development records only; observational, not causal',
        'new_training_calls': 0, 'new_inference_calls': 0, 'new_solver_calls': 0, 'test_read': False,
        'source_hashes_before_and_after': before_hashes, 'audit_source_sha256': sha(Path(__file__)),
        'numeric_thread_limit': 4, 'model_code_parity_E25_E42': parity,
        'max_abs_move_unit_difference': max(abs(r['raw_area_vs_input_area_move_difference']) for r in table),
        'E39_cohorts': groups, 'E39_within_n_associations': associations,
        'E42_scalar_confirmation': control42, 'E35_trajectories': trajectories,
        'examples': examples, 'seconds': time.perf_counter() - started}
    save(OUT / 'summary.json', summary)
    for name, rows in (('E39_retention.csv', table), ('E42_scalar_confirmation.csv', scalar42)):
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'seconds': summary['seconds'], 'E39_all': groups['all'],
                      'E42_move': control42['center_move'], 'parity': parity}, ensure_ascii=False))


if __name__ == '__main__':
    main()
