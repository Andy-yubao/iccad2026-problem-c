"""Descriptive balanced-panel and worker-grouped explanatory statistics, no draft training."""
import numpy as np
from scipy.stats import rankdata, spearmanr
from .case_features import PUBLIC, REFERENCE


def rho(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(spearmanr(x, y).statistic)


def conditional_rank_rho(x, y, n):
    a = np.zeros(len(x)); b = np.zeros(len(x))
    for count in np.unique(n):
        ids = np.flatnonzero(n == count)
        a[ids] = rankdata(np.asarray(x)[ids])- (len(ids)+1)/2
        b[ids] = rankdata(np.asarray(y)[ids])- (len(ids)+1)/2
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a @ b/denominator) if denominator > 1e-12 else None


def summary(values):
    values = np.asarray(values, float)
    return {'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99)), 'max': float(values.max()),
            'worst10pct_loss_sum_share': float(np.sort(values)[-max(1, int(np.ceil(.1*len(values)))):].sum()/values.sum())}


def panel_decomposition(matrix):
    matrix = np.asarray(matrix)
    grand = matrix.mean(); case = matrix.mean(1); model = matrix.mean(0)
    interaction = matrix-case[:, None]-model[None]+grand
    ss = {'case': matrix.shape[1]*np.square(case-grand).sum(),
          'model': matrix.shape[0]*np.square(model-grand).sum(), 'interaction': np.square(interaction).sum()}
    total = sum(ss.values())
    return {k: float(v/total) for k, v in ss.items()}


def concordance(loss, quality, workers, *, seed=69):
    # Equal case weight for the mean within-case pair accuracy; ties separately reported.
    pairs = np.triu_indices(loss.shape[1], 1)
    dl = loss[:, pairs[0]]-loss[:, pairs[1]]
    dq = quality[:, pairs[0]]-quality[:, pairs[1]]
    valid = (abs(dl) > 1e-9) & (abs(dq) > 1e-9)
    numerator = ((dl*dq > 0) & valid).sum(1); denominator = valid.sum(1)
    included = denominator > 0
    case_score = numerator[included]/denominator[included]
    groups = [np.flatnonzero(np.asarray(workers) == value) for value in np.unique(workers)]
    rng = np.random.default_rng(seed); draws = []
    for _ in range(500):
        ids = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        keep = denominator[ids] > 0
        if keep.any():
            draws.append(np.mean(numerator[ids][keep]/denominator[ids][keep]))
    return {'non_tied_pairs': int(valid.sum()), 'ties_or_near_ties': int((~valid).sum()),
            'cases_with_comparison': int(included.sum()), 'case_equal_weight_accuracy': float(case_score.mean()),
            'worker_bootstrap95': np.percentile(draws, [2.5, 97.5]).tolist(),
            'pooled_pair_accuracy': float(numerator.sum()/denominator.sum())}


def ridge_worker_cv(features, y, names, *, regularization=10.):
    # Fixed ridge/log-G explanatory model, not a trained layout network or a test score.
    x = np.asarray([[row[name] for name in names] for row in features], float)
    target = np.log(np.maximum(y, 1e-8)); groups = np.array([row['worker'] for row in features])
    unique = np.unique(groups); np.random.default_rng(69).shuffle(unique)
    folds = {worker: i % 5 for i, worker in enumerate(unique)}
    prediction = np.zeros(len(target))
    for fold in range(5):
        test = np.array([folds[group] == fold for group in groups]); train = ~test
        mean = x[train].mean(0); std = x[train].std(0); std[std < 1e-8] = 1.
        a = (x[train]-mean)/std; b = (x[test]-mean)/std; center = target[train].mean()
        beta = np.linalg.solve(a.T @ a+regularization*np.eye(a.shape[1]), a.T @ (target[train]-center))
        prediction[test] = b @ beta+center
    return {'target': 'log G', 'features': names, 'folds': 5, 'group': 'source worker', 'ridge_lambda': regularization,
            'out_of_fold_R2': float(1-np.square(target-prediction).sum()/np.square(target-target.mean()).sum()),
            'out_of_fold_rank_correlation': rho(target, prediction),
            'scope': 'Internal explanatory worker-held-out CV on exposed dev; no independent test or causal claim.'}


def analyze(features, records, reference, terminal, weights):
    names = list(records); n = np.array([row['n'] for row in features]); workers = np.array([row['worker'] for row in features])
    loss = np.column_stack([[row['loss_total'] for row in records[name]] for name in names])
    overlap = np.column_stack([[row['overlap_area_fraction'] for row in records[name]] for name in names])
    reference_g = np.array([row['total'] for row in reference])
    distributions = {name: summary(loss[:, i]) for i, name in enumerate(names)}
    correlations = [[rho(loss[:, i], loss[:, j]) for j in range(len(names))] for i in range(len(names))]
    hardest_sets = [set(np.argsort(loss[:, i])[-200:]) for i in range(len(names))]
    jaccard = [[len(a & b)/len(a | b) for b in hardest_sets] for a in hardest_sets]
    e65_index = names.index('E65_geometry'); e42_index = names.index('E42')
    y = loss[:, e65_index]
    rank = np.argsort(y); bottom = rank[:1000]; top = rank[-200:]
    terms = {}
    for label, ids in (('all', np.arange(len(y))), ('bottom50pct', bottom), ('top10pct', top)):
        values = {term: float(np.mean([records['E65_geometry'][i]['loss_'+term] for i in ids])*weight) for term, weight in weights.items()}
        terms[label] = {'count': len(ids), 'G_mean': float(y[ids].mean()), 'weighted_terms': values,
                        'term_shares': {k: v/y[ids].mean() for k, v in values.items()}}
    feature_correlations = []
    for name in [*PUBLIC, *REFERENCE]:
        x = np.array([row[name] for row in features])
        value = rho(x, y)
        if value is None:
            continue
        feature_correlations.append({'feature': name, 'uses_reference_label': name in REFERENCE,
              'spearman': value, 'within_exact_N_rank_rho': conditional_rank_rho(x, y, n),
              'low_loss_half_mean': float(x[bottom].mean()), 'high_loss_decile_mean': float(x[top].mean())})
    feature_correlations.sort(key=lambda row: abs(row['within_exact_N_rank_rho'] or 0), reverse=True)
    strata = {}
    for lo, hi in ((21, 39), (40, 59), (60, 79), (80, 99), (100, 120)):
        ids = np.flatnonzero((n >= lo) & (n <= hi))
        if len(ids):
            strata[f'{lo}-{hi}'] = {name: {'count': len(ids), 'G_mean': float(loss[ids, j].mean()),
                'overlap_mean': float(overlap[ids, j].mean())} for j, name in enumerate(names)}
    modules = {}
    module_keys = [key for key in records['E65_geometry'][0] if key.endswith(('_center_loss_share', '_center_error_global', '_center_error_local'))]
    module_keys += ['mean_center_error_global', 'mean_center_error_local', 'translation_fraction_center_mse',
                    'center_loss_after_remove_mean_shift', 'top10pct_modules_center_loss_share']
    for label, ids in (('all', np.arange(len(y))), ('bottom50pct', bottom), ('top10pct', top)):
        modules[label] = {key: float(np.mean([records['E65_geometry'][i][key] for i in ids])) for key in module_keys}
    diagnostics = {'reference_state_G': summary(reference_g), 'reference_state_G_vs_E65_G_rho': rho(reference_g, y),
                   'E65_center_loss_mean': float(np.mean([row['loss_center_local'] for row in records['E65_geometry']])),
                   'reference_state_caveat': 'Not an optimum or lower bound; original centers/log-ratios decoded with transformed public areas.'}
    model_prediction = {
        'N_only': ridge_worker_cv(features, y, ['log_n', 'log_n_squared']),
        'public_inputs': ridge_worker_cv(features, y, PUBLIC),
        'public_and_reference': ridge_worker_cv(features, y, [*PUBLIC, *REFERENCE])}
    cross_case = {name: {'G_vs_overlap_rho': rho(loss[:, j], overlap[:, j]),
                        'G_vs_overlap_within_exact_N': conditional_rank_rho(loss[:, j], overlap[:, j], n)} for j, name in enumerate(names)}
    # Saved flash500 is a nested panel; not another independent sample.
    indices = terminal['indices']; q = terminal['Q']; mov = terminal['movement']
    for j, name in enumerate(names):
        cross_case[name].update(G_vs_flash_Q_rho=rho(loss[indices, j], q[:, j]),
             G_vs_flash_Q_within_exact_N=conditional_rank_rho(loss[indices, j], q[:, j], n[indices]),
             G_vs_repair_movement_rho=rho(loss[indices, j], mov[:, j]))
    dl = loss[:, e65_index]-loss[:, e42_index]
    do = overlap[:, e65_index]-overlap[:, e42_index]
    dq = q[:, e65_index]-q[:, e42_index]
    paired = {'E65_minus_E42': {'G_delta_vs_overlap_delta_rho': rho(dl, do),
        'G_better_overlap_worse_count': int(((dl < -1e-9) & (do > 1e-9)).sum()),
        'G_better_count': int((dl < -1e-9).sum()), 'total': len(dl),
        'G_delta_vs_Q_delta_rho_dev500': rho(dl[indices], dq),
        'G_better_Q_worse_count_dev500': int(((dl[indices] < -1e-9) & (dq > 1e-9)).sum())}}
    return {'models': names, 'case_count': len(y), 'worker_count': len(np.unique(workers)), 'distributions': distributions,
        'balanced_panel_variation_raw_G': panel_decomposition(loss),
        'balanced_panel_variation_log_G': panel_decomposition(np.log(np.maximum(loss, 1e-8))),
        'Loss_rank_correlations': correlations, 'worst10pct_Jaccard': jaccard,
        'E65_weighted_term_decomposition': terms, 'E65_feature_associations': feature_correlations,
        'N_strata': strata, 'E65_module_diagnostics': modules, 'reference_state_diagnostic': diagnostics,
        'explanatory_ridge_worker_CV': model_prediction, 'cross_case_correlations': cross_case,
        'same_case_ranking_overlap_dev2000': concordance(loss, overlap, workers),
        'same_case_ranking_flash_dev500': concordance(loss[indices], q, workers[indices]),
        'paired_changes': paired, 'test_read': False, 'training_started': False}
