"""Within-layout decoded-center associations, using saved500 drafts only."""
import os
import numpy as np


def case_nodes(payload):
    worker, area, constraints, features, reference, predicted = payload
    c = np.asarray(constraints); mutable = c[:, 1] == 0
    size = np.sqrt(area)
    pin = np.asarray(features)[:, 14] > 0
    logarea = np.log(np.maximum(area, 1e-12))
    x = np.column_stack((pin, logarea, logarea**2, np.asarray(features)[:, 13], c[:, 4] > 0, c[:, 0] != 0)).astype(float)[mutable]
    x -= x.mean(0)
    n = len(x)
    ref_center = reference[:, :2]+reference[:, 2:]/2
    output = {'worker': worker, 'pid': os.getpid(), 'xx': (x.T @ x/max(n, 1)).tolist(), 'models': {}}
    for name, positions in predicted.items():
        center = positions[:, :2]+positions[:, 2:]/2
        error = np.linalg.norm(center-ref_center, axis=-1)/size
        y = np.log1p(error[mutable]); y -= y.mean()
        a = error[mutable & pin]; b = error[mutable & ~pin]
        output['models'][name] = {'xy': (x.T @ y/max(n, 1)).tolist(),
            'pin_count': len(a), 'no_pin_count': len(b), 'pin_error_sum': float(a.sum()),
            'no_pin_error_sum': float(b.sum()),
            'mixed_case_difference': float(a.mean()-b.mean()) if len(a) and len(b) else None}
    return output


def summarize_nodes(records):
    groups = {}
    for row in records:
        groups.setdefault(row['worker'], []).append(row)
    workers = list(groups); xx = sum(np.asarray(row['xx']) for row in records)
    result = {'case_count': len(records), 'worker_count': len(workers), 'worker_pids': sorted({r['pid'] for r in records}),
              'definition': 'Saved decoded center Euclidean error /sqrt(public effective area), not raw pre-clamp center Loss',
              'regression': 'Case-fixed-effect OLS of log1p(local error), equal case weight; controls log(area),log(area)^2,log(weighted degree),boundary flag,fixed-shape flag',
              'coefficient_order': ['pin_connected', 'log_area', 'log_area_squared', 'log_weighted_degree', 'boundary', 'fixed_shape'],
              'models': {}}
    for name in records[0]['models']:
        xy = sum(np.asarray(row['models'][name]['xy']) for row in records)
        beta = np.linalg.solve(xx+1e-9*np.eye(len(xy)), xy)
        rng = np.random.default_rng(690); draws = []
        blocks = []
        for worker in workers:
            rows = groups[worker]
            blocks.append((sum(np.asarray(row['xx']) for row in rows),
                           sum(np.asarray(row['models'][name]['xy']) for row in rows)))
        for _ in range(500):
            ids = rng.integers(len(blocks), size=len(blocks))
            a = sum(blocks[j][0] for j in ids); b = sum(blocks[j][1] for j in ids)
            draws.append(np.linalg.solve(a+1e-9*np.eye(len(b)), b)[0])
        values = [row['models'][name] for row in records]
        a_count = sum(row['pin_count'] for row in values); b_count = sum(row['no_pin_count'] for row in values)
        differences = np.array([row['mixed_case_difference'] for row in values if row['mixed_case_difference'] is not None])
        result['models'][name] = {'coefficients': beta.tolist(), 'pin_coefficient_worker_bootstrap95': np.percentile(draws, [2.5,97.5]).tolist(),
             'pin_modules': a_count, 'no_pin_modules': b_count,
             'pin_mean_decoded_local_error': sum(row['pin_error_sum'] for row in values)/a_count,
             'no_pin_mean_decoded_local_error': sum(row['no_pin_error_sum'] for row in values)/b_count,
             'mixed_case_count': len(differences), 'mixed_case_mean_error_difference': float(differences.mean()),
             'mixed_cases_pin_error_lower_fraction': float((differences < 0).mean())}
    return result
