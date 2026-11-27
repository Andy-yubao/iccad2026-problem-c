"""Saved-output follow-up associations; no forward, repair, training or test read."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from .io import HERE, save, read
from .case_features import PUBLIC
from .statistics import ridge_worker_cv
from .node_analysis import case_nodes, summarize_nodes

OUT = HERE/'runs/20261028_dev2000'


def main():
    from .data import load_rows
    from .collect import RUNS, key
    features = read(OUT/'features.json')['records']
    forward = read(OUT/'forwards.json')
    y = np.array([row['loss_total'] for row in forward['models']['E65_geometry']])
    pins = ['pin_edges_per_node', 'pin_count_per_node', 'pin_connected_fraction', 'pin_weight_cv']
    graph = ['connection_density', 'mean_degree', 'degree_cv', 'weighted_degree_cv',
             'isolated_fraction', 'graph_components_fraction', 'cross_group_edge_fraction']
    predictions = {'N_plus_pin_summaries': ridge_worker_cv(features, y, ['log_n','log_n_squared',*pins]),
                   'public_without_pin_summaries': ridge_worker_cv(features, y, [name for name in PUBLIC if name not in pins]),
                   'public_without_connectivity_summaries': ridge_worker_cv(features, y, [name for name in PUBLIC if name not in graph])}
    # Multivariable partial rank association after explicit nuisance adjustment.
    from scipy.stats import rankdata
    controls = ['log_n','log_n_squared','connection_density','preplaced_fraction','fixed_fraction','area_gini','boundary_fraction']
    x = np.column_stack((np.ones(len(features)), *[rankdata([row[name] for row in features]) for name in controls]))
    target = rankdata(y); pin = rankdata([row['pin_connected_fraction'] for row in features])
    residual_y = target-x @ np.linalg.lstsq(x, target, rcond=None)[0]
    residual_pin = pin-x @ np.linalg.lstsq(x, pin, rcond=None)[0]
    partial = float(np.corrcoef(residual_y, residual_pin)[0,1])
    # Saved500 decoded drafts provide a within-case association without another forward.
    rows = load_rows('dev'); mapping = {key(row['member']): row for row in rows}
    # Historical E42 terminal records omit raw coordinates; use the complete saved E65 panel.
    terminal = {'E65': read(RUNS['E65_geometry']/'terminal/results.json')['records']}
    predictions_by_member = {name: {key(row['member']): row for row in data} for name,data in terminal.items()}
    payloads = []
    for member, record in predictions_by_member['E65'].items():
        row = mapping[member]; g = row['graph']; scale = float(g['scale'])
        payloads.append((member.split('/')[0], g['area'].numpy()*scale**2,
                        g['constraints'].numpy(), g['features'].numpy(), g['reference'].numpy()*scale,
                        {name: np.asarray(records[member]['draft_positions']) for name,records in predictions_by_member.items()}))
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        node_records = list(pool.map(case_nodes, payloads, chunksize=8))
    result = {'case_feature_CV_ablations': predictions,
              'pin_coverage_partial_rank_rho': partial, 'partial_rank_controls': controls,
              'within_case_node_associations': summarize_nodes(node_records),
              'scope': 'Post-hoc exploratory follow-up to initial feature screen; no causal proof or draft-network training',
              'new_inference': False, 'test_read': False}
    save(OUT/'followup.json', result)
    print('Saved-output follow-up:', {name: row['out_of_fold_R2'] for name,row in predictions.items()}, flush=True)
    print('Pin partial rank rho:', partial, flush=True)
    print('Within-case node associations:', result['within_case_node_associations']['models'], flush=True)


if __name__ == '__main__':
    main()
