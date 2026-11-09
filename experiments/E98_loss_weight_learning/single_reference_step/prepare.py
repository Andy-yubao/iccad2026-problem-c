"""Freeze one direction and a fresh train100/1000 panel before observing results."""
import numpy as np
from ..io import ROOT,HERE,sha,save
from ..loss_config import COEFFICIENTS
from ..dataset_view import ReferenceLayouts
from .common import OWNER,PLAN_PATH,read,MODEL

def main():
    if PLAN_PATH.exists():raise RuntimeError('Plan already registered; do not replace')
    parent=HERE/'runs/20261104_generalization96_seed0/checkpoints/best.pt'
    parent_eval=parent.parent.parent/'evaluation_depth_dev500'
    frozen=read(parent_eval/'frozen.json')
    assert sha(parent)==frozen['checkpoint_sha256']=='3f7ca0086af8a13bd52eedf0635117312e5c0a72e80e423e9cee3db5ca9b87ea'
    assert read(parent.parent.parent/'results.json')['best_epoch']==71
    assert read(parent_eval/'results.json')['count']==500
    base=dict(COEFFICIENTS)
    candidate=dict(base)
    for name in ('position','shape'):candidate[name]/=1.2
    records=ReferenceLayouts('train').records
    assert len(records)==16000
    pool_seed,panel_seed,sampler_seed=2026110411,2026110412,2026110413
    pool=np.random.default_rng(pool_seed).choice(16000,1000,replace=False).tolist()
    panel=np.random.default_rng(panel_seed).choice(pool,100,replace=False).tolist()
    plan=dict(revision='E98.single_reference120.from_generalization_best71.v1',
        authorization='User: continue only one step weakening position and shape reference, inspect improvement; discard previous search results. Single candidate training plus train100/1000 comparison and frozen terminal dev500; no other directions.',
        parent_checkpoint=parent.relative_to(ROOT).as_posix(),parent_sha256=sha(parent),
        parent_best_epoch=71,parent_evaluation=parent_eval.relative_to(ROOT).as_posix(),
        parent_frozen_evaluation_sha256=sha(parent_eval/'frozen.json'),
        parent_evaluation_results_sha256=sha(parent_eval/'results.json'),
        initial_coefficients=base,candidate_coefficients=candidate,
        coefficient_change='position1/1.2 and shape.25/1.2 only; other8 coefficients bitwise unchanged; NO total-sum normalization',
        architecture=MODEL,initialization='same frozen generalization best71 weights and RNG; fresh AdamW m/v/age for candidate; frozen baseline has zero updates',
        previous_search='all old pilot/dynamic results excluded from weights, coefficients, direction choice, and acceptance; old evidence retained unchanged',
        outer_rounds=1,candidate_count=1,other_directions=False,automatic_next_stage=False,
        train_count=16000,train_monitor_count=500,dev_monitor_count=500,
        data='original v5 train16k; original mean-area MIB task; frozen splits and fixed500 monitors unchanged',
        batch_size=32,precision='FP32 deterministic, no AMP/TF32',
        augmentation='same .10 FFN/.05 residual dropout and .5 exact xy axis-swap including fixed/preplaced specs; epoch/member flags independent of Dropout',
        optimizer='fresh AdamW decay.0003, clip1, relative module update bound.005; retain complete state within-phase resume',
        initial_lr=1e-4,lower_lr=5e-5,protection_epochs=2,patience=3,
        earliest_drop_after_epoch=5,lower_lr_stall_epochs=3,maximum_epochs=15,
        selection='same own fixed Loss dev500 mean for best/LR/earlystop; epoch0 eligible; no Q checkpoint selection',
        feedback_pool_seed=pool_seed,feedback_panel_seed=panel_seed,sampler_seed=sampler_seed,
        feedback_pool_train_indices=pool,feedback_pool_members=[records[i]['member'] for i in pool],
        feedback_train_indices=panel,feedback_members=[records[i]['member'] for i in panel],
        feedback_cases=100,feedback_pool_cases=1000,feedback_repeats=1,
        feedback_objective='mixed_Q=.5*(weighted_Q+mean_Q); report all3; same current100 for baseline/candidate; hard legal required',
        minimum_gain=.0001,evaluator='unchanged E98.depth_budget.v1; one solve per unique model/input; no clock-based quality decisions',
        terminal='candidate own-Loss best frozen before Q; once dev500, baseline completed500 reused on same depth/model-input/scoring sources; same100 images; report even if train feedback does not improve, never use devQ to launch more training',
        interpretation='one adaptive development comparison after baseline devQ exposure; gains can include extra training; total Loss across changed coefficients not directly comparable',
        implementation_train_indices=[1683,10027,5474,10140,15001],test_read=False,
        stop='run/STOP stops accepted batch boundary, saves latest, blocks all later feedback/dev stages',
        automatic_wakeups=False)
    save(PLAN_PATH,plan)
    print('One candidate registered:',candidate['position'],candidate['shape'],'pool1000/panel100',flush=True)

if __name__=='__main__':main()
