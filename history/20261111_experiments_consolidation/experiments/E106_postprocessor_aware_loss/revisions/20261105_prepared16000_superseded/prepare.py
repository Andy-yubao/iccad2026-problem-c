"""Register exactly 2000 train problems x 8 drafts, never load dev/test payloads."""
from collections import Counter
import numpy as np
from datasets.reference_layouts import ReferenceLayouts
from .common import ROOT, OWNER, RUN, read, save, sha


def main():
    if (OWNER / 'panel.json').exists():
        raise RuntimeError('Frozen panel already exists; no silent resampling')
    rows = ReferenceLayouts('train').records
    old = ROOT / 'experiments/E103_offline_loss_alignment'
    exclude = set(read(old / 'panel.json')['train_indices'])
    exclude.update(r['train_index'] for r in read(old / 'global_loss_q/panel.json')['samples'])
    rng = np.random.default_rng(2026110506)
    candidates = np.array([i for i in range(16000) if i not in exclude])
    selected = rng.choice(candidates, 2000, replace=False).tolist()
    assignment = rng.permutation(2000)
    roles = {int(i): 'fit' if j < 1200 else 'validation' if j < 1600 else 'check'
             for j, i in enumerate(assignment)}
    secondary = rng.permutation(np.repeat(['learned29', 'reference075_8'], 1000))
    cases = [dict(index=j, train_index=i, member=rows[i]['member'], n=rows[i]['n'],
                  split=roles[j], secondary_model=str(secondary[j]), perturb_seed=2026110506000+j)
             for j, i in enumerate(selected)]
    panel = dict(revision='E106.train2000.drafts16000.v1', cases=cases,
                 total_problems=2000, total_drafts=16000, drafts_per_problem=8,
                 split_counts=dict(Counter(roles.values())), sampling_seed=2026110506,
                 n_counts=dict(Counter(r['n'] for r in cases)), excluded_old_train_cases=len(exclude),
                 source_partition='original train16000 only', test_read=False, payloads_read=0)
    save(OWNER / 'panel.json', panel)
    plan = dict(
        revision='E106.postprocessor_aware_loss.v1', status='implementation_and_data_preparation_only',
        authorization='New independent experiment; Scheme A; TOTAL 16000 drafts; implement but DO NOT START TRAINING',
        training_authorized=False, layout_training_authorized=False, automatic_training=False,
        data=dict(problems=2000, drafts_per_problem=8, total_drafts=16000,
                  original_drafts=4000, perturbed_drafts=12000, pairs_per_problem=3,
                  split_problems=dict(fit=1200, validation=400, check=400),
                  split_drafts=dict(fit=9600, validation=3200, check=3200),
                  all_related_drafts_same_partition=True, network_unseen_claim=False,
                  source_panel_sha256=sha(OWNER/'panel.json'), old_Q_or_drafts_reused=False),
        perturbations=dict(primary='enhanced71', secondary='1000 problems learned29;1000 reference075_8',
                           pairs=['primary normalized Gaussian .005/.020',
                                  'primary structural .020: group/boundary/cluster/shape',
                                  'secondary normalized Gaussian .010/.030'],
                           hard_masks='fixed shape only locks aspect; preplaced locks all3 state channels',
                           step_unit='sqrt(public module area) for centers,1 for log-aspect; direction active RMS1',
                           derivatives='finite-radius symmetric secants, NOT unbiased black-box gradient claims'),
        model=dict(width=256, layers=12, heads=8, hidden=768, pooling_queries=4,
                   ffn_dropout=.10, residual_dropout=.05,
                   design='Full graph+decoded geometry, pair attention bias, separately pre-normalized wire/MIB/group branches, gated residuals, SwiGLU, mean/max/attention pooling',
                   input='public features, legal locked specs, complete current draft; no reference, targets, denominator, sourceID or problemID'),
        objective=dict(value='raw Q standardized squared error',
                       directions='three symmetric Q slopes standardized squared error; related drafts share dropout',
                       auxiliary='signed HPWLgap, signed area gap, soft-violation ratio',
                       coefficients=dict(value=1., directions=1., auxiliary=.1),
                       normalization='fit1200 labels only; saved once after bank complete',
                       case_weight='.5+.5*exp((N-120)/12)/role_mean_exp; role mean fixed over whole panel, never batch-renormalized',
                       same_function_for_train_monitor_best_lr_earlystop=True),
        augmentation=dict(xy_swap=False, reason='Fixed E98 heuristic is not proven XY-equivariant; transformed graph requires new true Q labels. No copying labels across XY swaps.'),
        proposed_training=dict(seed=0, problems_per_batch=4, drafts_per_batch=32, precision='FP32',
                               optimizer='AdamW', lr_tiers=[3e-4,1e-4,3e-5], weight_decay=.001,
                               clip_norm=1., plateau_patience=5, lowest_tier_patience=8,
                               meaningful_absolute=1e-5, meaningful_relative=.001,
                               maximum_epochs=None, maximum_seconds=None,
                               best_selection='fixed internal validation400 composite fitting Loss only; no dev500 or Q re-selection'),
        bank=dict(workers=4, cpu_threads_per_worker=1, gpu=False, maximum_solver_calls=16000,
                  postprocessor='Unmodified copied E98.depth_budget.v1', test_read=False,
                  terminal='Save data, numerical summaries and stop. NEVER invoke training.'),
        preflight=dict(maximum_unique_cases=5, optimizer_updates=0, original_check_partition_untouched=True),
        old_experiments='No old sources, weights, results, STOPs or active runs modified',
    )
    save(OWNER / 'plan.json', plan)
    save(RUN / 'preparation_state.json', dict(status='prepared_not_training', total_drafts=16000,
                                             training_authorized=False, test_read=False))
    print('Registered2000 train problems,16000 TOTAL drafts; training remains disabled.')


if __name__ == '__main__':
    main()
