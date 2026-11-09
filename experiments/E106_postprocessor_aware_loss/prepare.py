"""Register5000 train problems x5 drafts, no dev/test payloads or training."""
from collections import Counter
import numpy as np
from datasets.reference_layouts import ReferenceLayouts
from .common import ROOT, OWNER, RUN, read, save, sha


def main():
    if (OWNER / 'panel.json').exists():
        prior = read(OWNER / 'panel.json')
        archived = OWNER / 'revisions/20261105_prepared16000_superseded/panel.json'
        if prior['total_drafts'] != 16000 or not archived.exists() or sha(archived) != sha(OWNER/'panel.json'):
            raise RuntimeError('Frozen panel exists; only explicitly archived user-requested revision allowed')
    rows = ReferenceLayouts('train').records
    old = ROOT / 'experiments/E103_offline_loss_alignment'
    exclude = set(read(old / 'panel.json')['train_indices'])
    exclude.update(r['train_index'] for r in read(old / 'global_loss_q/panel.json')['samples'])
    rng = np.random.default_rng(2026110506)
    candidates = np.array([i for i in range(16000) if i not in exclude])
    selected = rng.choice(candidates, 5000, replace=False).tolist()
    assignment = rng.permutation(5000)
    roles = {int(i): 'fit' if j < 3000 else 'validation' if j < 4000 else 'check'
             for j, i in enumerate(assignment)}
    cases = [dict(index=j, train_index=i, member=rows[i]['member'], n=rows[i]['n'],
                  split=roles[j], perturb_seed=2026110506000+j)
             for j, i in enumerate(selected)]
    panel = dict(revision='E106.train5000.drafts25000.v2', cases=cases,
                 total_problems=5000, total_drafts=25000, drafts_per_problem=5,
                 split_counts=dict(Counter(roles.values())), sampling_seed=2026110506,
                 n_counts=dict(Counter(r['n'] for r in cases)), excluded_old_train_cases=len(exclude),
                 source_partition='original train16000 only', test_read=False, payloads_read=0)
    save(OWNER / 'panel.json', panel)
    plan = dict(
        revision='E106.postprocessor_aware_loss.v2', status='implementation_and_data_preparation_only',
        authorization='New independent experiment; Scheme A; latest5000problems x5drafts=25000; about2Mparameters; implement but DO NOT START TRAINING',
        training_authorized=False, layout_training_authorized=False, automatic_training=False,
        data=dict(problems=5000, drafts_per_problem=5, total_drafts=25000,
                  original_drafts=15000, perturbed_drafts=10000, pairs_per_problem=1,
                  split_problems=dict(fit=3000, validation=1000, check=1000),
                  split_drafts=dict(fit=15000, validation=5000, check=5000),
                  all_related_drafts_same_partition=True, network_unseen_claim=False,
                  source_panel_sha256=sha(OWNER/'panel.json'), old_Q_or_drafts_reused=False),
        perturbations=dict(centers=['enhanced71','learned29','reference075_8'],
                           pairs=['one symmetric pair per problem; roots rotate across3 models; half Gaussian half structural; radii.005/.010/.020/.030 balanced deterministically'],
                           hard_masks='fixed shape only locks aspect; preplaced locks all3 state channels',
                           step_unit='sqrt(public module area) for centers,1 for log-aspect; direction active RMS1',
                           derivatives='finite-radius symmetric secants, NOT unbiased black-box gradient claims'),
        model=dict(width=128, layers=8, heads=4, hidden=320, pooling_queries=2, readout_width=256,
                   ffn_dropout=.10, residual_dropout=.05,
                   design='Full graph+decoded geometry, pair attention bias, separately pre-normalized wire/MIB/group branches, gated residuals, SwiGLU, mean/max/attention pooling',
                   input='public features, legal locked specs, complete current draft; no reference, targets, denominator, sourceID or problemID'),
        objective=dict(value='raw Q standardized squared error',
                       directions='one symmetric Q slope standardized squared error; related drafts share dropout',
                       auxiliary='signed HPWLgap, signed area gap, soft-violation ratio',
                       coefficients=dict(value=1., directions=1., auxiliary=.1),
                       normalization='fit3000 labels only; saved once after bank complete',
                       case_weight='.5+.5*exp((N-120)/12)/role_mean_exp; role mean fixed over whole panel, never batch-renormalized',
                       same_function_for_train_monitor_best_lr_earlystop=True),
        augmentation=dict(xy_swap=False, reason='Fixed E98 heuristic is not proven XY-equivariant; transformed graph requires new true Q labels. No copying labels across XY swaps.'),
        proposed_training=dict(seed=0, problems_per_batch=6, drafts_per_batch=30, precision='FP32',
                               optimizer='AdamW', lr_tiers=[3e-4,1e-4,3e-5], weight_decay=.001,
                               clip_norm=1., plateau_patience=5, lowest_tier_patience=8,
                               meaningful_absolute=1e-5, meaningful_relative=.001,
                               maximum_epochs=None, maximum_seconds=None,
                               best_selection='fixed internal validation1000 composite fitting Loss only; no dev500 or Q re-selection'),
        bank=dict(workers=4, cpu_threads_per_worker=1, gpu=False, maximum_solver_calls=25000,
                  postprocessor='Unmodified copied E98.depth_budget.v1', test_read=False,
                  terminal='Save data, numerical summaries and stop. NEVER invoke training.'),
        preflight=dict(maximum_unique_cases=5, optimizer_updates=0, original_check_partition_untouched=True),
        old_experiments='No old sources, weights, results, STOPs or active runs modified',
    )
    save(OWNER / 'plan.json', plan)
    save(OWNER / 'authorization.json', dict(critic_training=False, layout_training=False,
                                           reason='User explicitly requested implementation WITHOUT training'))
    save(RUN / 'preparation_state.json', dict(status='prepared_not_training', total_drafts=25000,
                                             training_authorized=False, test_read=False))
    print('Registered5000 train problems,25000 TOTAL drafts; training remains disabled.')


if __name__ == '__main__':
    main()
