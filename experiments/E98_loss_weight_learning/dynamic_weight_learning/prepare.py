"""Freeze all directions and train-only panels before any outcome is observed."""
import math
import torch
import numpy as np
from ..loss_config import COEFFICIENTS
from ..dataset_view import ReferenceLayouts
from ..io import ROOT, HERE, sha, save
from .common import RUN, PLAN_PATH, MODEL, KEYS, cache

def main():
    if PLAN_PATH.exists():
        raise RuntimeError('Plan already exists; do not replace the frozen registration')
    parent = HERE/'runs/20261102_fixed_rich96_seed0/checkpoints/best.pt'
    digest = sha(parent)
    assert digest == '1df099d624797d3e61fd332b7ffafed8ebc213f73758649e2a34054b395201d1'
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    assert saved['model_config'] == MODEL and saved['progress']['epoch'] == 58
    baseline_algorithms = {}
    for name in ('model.py','features.py','batches.py','ports.py','loss.py','simple_loss.py',
                 'loss_config.py','group_tree.py','captured_projection.py','step_control.py'):
        path = HERE/name
        key = path.relative_to(ROOT).as_posix()
        assert sha(path) == saved['identity']['sources'][key], ('Baseline algorithm drift',key)
        baseline_algorithms[key] = sha(path)
    cache.verify('train'); cache.verify('dev')
    records = ReferenceLayouts('train').records
    assert len(records) == 16000
    pool_seed, panel_seed, direction_seed = 2026110401, 2026110402, 2026110403
    pool = np.random.default_rng(pool_seed).choice(16000, 1000, replace=False).tolist()
    designed = [
        ('reference_vs_public_geometry', [-1,-1,1,1,1,1,1,1,1,1]),
        ('overlap_vs_wire_compactness', [0,0,1,1,1,-1,-1,0,0,0]),
        ('worst_overlap_vs_mean_overlap', [0,0,-1,-1,1,0,0,0,0,0]),
        ('wirelength_vs_compactness', [0,0,0,0,0,1,-1,0,0,0]),
        ('boundary_vs_group_contact', [0,0,0,0,0,0,0,1,-1,0]),
    ]
    rng = np.random.default_rng(direction_seed)
    directions = [(name, d, 'designed') for name,d in designed]
    for i in range(5):
        while True:
            d = rng.choice([-1,1], len(KEYS)).tolist()
            if len(set(d)) > 1 and not any(d == old or d == [-x for x in old] for _,old,_ in directions):
                break
        directions.append((f'random_{i+1:02d}', d, 'seeded_random'))
    rounds = []
    for i,(name,d,kind) in enumerate(directions):
        panel = np.random.default_rng(panel_seed+i).choice(pool,100,replace=False).tolist()
        rounds.append(dict(round=i+1,name=name,kind=kind,direction=d,
                           sampler_seed=2026110500+i,feedback_train_indices=panel,
                           feedback_members=[records[j]['member'] for j in panel]))
    plan = dict(revision='E98.dynamic_mixedq.dev_loss_earlystop.v1',
        authorization='User explicitly authorized preparation and background multi-round search; five designed groups followed by seeded random groups; original fixed baseline restart; train-only branch acceptance 100/1000; validation Loss scheduling and early-stop; test sealed.',
        initial_checkpoint=parent.relative_to(ROOT).as_posix(), initial_sha256=digest,
        old_pilot='Excluded from initialization, direction design, acceptance and conclusions; evidence retained unmodified, not deleted.',
        architecture=MODEL, initial_coefficients=dict(COEFFICIENTS), coefficient_order=list(KEYS),
        log_perturbation=math.log(1.25), coefficient_rule='positive multiplicative +/- direction, preserve total coefficient sum; zero direction components are unperturbed before common normalization',
        rounds=rounds, maximum_outer_rounds=10,
        outer_budget='five designed directions then five random directions; stop after round10; no automatic wakeups or extension',
        train_count=16000, train_monitor_count=500, dev_monitor_count=500,
        data='original v5 16k; validated v6 prefix cache; original public MIB mean-area task; frozen split and monitors unchanged',
        feedback_pool_seed=pool_seed, feedback_panel_seed=panel_seed, direction_seed=direction_seed,
        feedback_pool_train_indices=pool, feedback_pool_members=[records[j]['member'] for j in pool],
        feedback_cases_per_round=100, feedback_repeats=2,
        sampling='uniform without replacement: fixed1000 from train16k; independently pre-frozen100 per round from that pool; same100 across arms/repeats',
        branch_arms=['control','plus','minus'], batch_size=32, precision='FP32; deterministic; no AMP/TF32',
        optimizer='fresh AdamW m/v/step for ALL three arms at each new phase, weight_decay.0003; global clip1 and per-module relative proposed-update bound.005 retained; exact optimizer/RNG/cursor/scheduler/projection preserved within-phase resume',
        initial_lr=1e-4, lower_lr=5e-5, protection_epochs=2, plateau_patience=3,
        scheduler_baseline_epoch=2, earliest_drop_after_epoch=5,
        minimum_meaningful_absolute=1e-5, minimum_meaningful_relative=.001,
        best_minimum_absolute=1e-5, lower_lr_stall_epochs=3, maximum_epochs_per_arm=10,
        checkpoint_selection='fixed own-Loss dev500 arithmetic mean best; same Loss in backward/train/dev monitoring; no Q checkpoint selection; independent branch early-stop means unequal actual compute',
        feedback_objective='mixed_Q=.5*official neutral-time weighted_Q+.5*arithmetic mean_Q; report all three; hard legality required',
        minimum_absolute_gain_per_repeat=.0001,
        acceptance='candidate mixed_Q beats control by >.0001 in BOTH repeats and all100 hard legal in both; choose lower repeat mean; tie keeps control; otherwise retain control; carry winning own-dev-Loss best model and coefficients to next round',
        acceptance_data='train only; dev Loss may select epochs and schedule, but dev postprocessed Q never selects coefficients; test payload forbidden',
        terminal='after round10 freeze unique model, once dev500 full repair and same100 reference/L/final gallery; no Q feedback or reselection',
        cache_path_migration='E42 prepared cache paths map to history/20261103_final_delivery_consolidation; original manifests,size,mtime,identities and feature hashes verified',
        baseline_algorithm_hashes=baseline_algorithms,
        cache_identity=cache.identity(), baseline_elapsed_seconds=saved['elapsed_seconds'],
        numerical_guards='original finite/Loss<1000/gradient<10000 before Adam, no skip or retry',
        implementation_shortcheck_indices=[1683,5474,10027,10140,15001],
        test_read=False, automatic_wakeups=False, stop='create run/STOP; batch boundary latest save; no later arms/rounds/terminal evaluation')
    save(PLAN_PATH, plan)
    save(RUN/'registration.json',dict(status='prepared',plan_sha256=sha(PLAN_PATH),
         baseline_sha256=digest,original_pilot_excluded=True,test_read=False))
    print('Prepared:',PLAN_PATH,'rounds',len(rounds),'pool',len(pool),'panels',len(rounds[0]['feedback_train_indices']),flush=True)

if __name__ == '__main__':
    main()
