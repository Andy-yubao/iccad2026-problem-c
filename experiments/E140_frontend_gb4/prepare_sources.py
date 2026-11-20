"""Prepare E139-matched early grouping/boundary x4; never launch training."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'experiments/E139_delayed_overlap'
ARM='e119_front_gb4'
MAPPING={'experiments.E139_delayed_overlap':'experiments.E140_frontend_gb4',
         'E139.delayed_overlap.v1.seed0':'E140.frontend_gb4.v1.seed0',
         'E139 training and original dev500 postprocessing completed.':'E140 training and original dev500 postprocessing completed.',
         "'# E139 '":"'# E140 '"}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Already prepared; preserve registered sources')
    gate=json.loads((PARENT/'preflight.json').read_text())
    for name,digest in gate['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen parent source drift: '+name)
    names=('__init__.py','augmentation.py','batches.py','captured_projection.py','cuda_step.py','data.py',
           'features.py','group_tree.py','ports.py','resident.py','schedule.py','simple_loss.py','step_control.py',
           'original_model.py','e119_model.py','model.py','io.py','train.py','run_queue.py','loss.py','preflight.py')
    copied={}
    files=[PARENT/name for name in names]+list((PARENT/'evaluation').rglob('*.py'))+[PARENT/'evaluation/budget.json']
    for source in files:
        name=source.relative_to(PARENT).as_posix();destination=HERE/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        original=source.read_text(encoding='utf-8');content=original
        for old,new in MAPPING.items():content=content.replace(old,new)
        if content==original:shutil.copyfile(source,destination)
        else:destination.write_text(content,encoding='utf-8')
        copied[name]=dict(parent=source.relative_to(ROOT).as_posix(),sha256=sha(source))
    loss=(HERE/'loss.py').read_text(encoding='utf-8')
    needle="    if layer in NO_OVERLAP_LAYERS:\n        # Omit"
    assert needle in loss
    loss=loss.replace(needle,"    if layer in NO_OVERLAP_LAYERS:\n        out['grouping_weighted'] = 4.0*out['grouping_weighted']\n        out['boundary_weighted'] = 4.0*out['boundary_weighted']\n        # Omit",1)
    (HERE/'loss.py').write_text(loss,encoding='utf-8')
    plan=json.loads((PARENT/'plan.json').read_text())
    parent_result=PARENT/'runs/e119_delayed_overlap_seed0/evaluation_no_mib_dev500/results.json'
    parent_weight=PARENT/'runs/e119_delayed_overlap_seed0/checkpoints/best.pt'
    controls=[dict(arm='e119_delayed_overlap',result=parent_result.relative_to(ROOT).as_posix(),
        results_sha256=sha(parent_result),checkpoint=parent_weight.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(parent_weight),best_epoch=42)]
    arm_plan=dict(plan['arms']['e119_delayed_overlap'],grouping_multipliers={'2':4,'6':4,'8':1,'10':1,'12':1},
                 boundary_multipliers={'2':4,'6':4,'8':1,'10':1,'12':1})
    plan.update(experiment='E140 early grouping/boundary x4 versus E139',revision='E140.frontend_gb4.v1.seed0',
        status='preparation_only',authorization='2026-11-15 user: prepare one E139-controlled experiment; enhance grouping/boundary at S2/S6 only; later three unchanged; no training launch requested',
        arms={ARM:arm_plan},initialization='scratch seed0; exact same initial tensors as E139; no warm start',
        loss='E139 aggregate + 0.15*(G2+B2+G6+B6), where G/B are original weighted per-layer contributions; S2/S6 overlap remains zero',
        comparison='Sole new training arm versus frozen E139 best42; only S2/S6 grouping/boundary coefficients change. Same seed/data/init/schedule/budgets; adaptive stopping may differ; single seed does not establish reliability.',
        comparison_controls=controls,new_candidates=1,topology_similarity_in_training=False,
        loss_coefficients_frozen=True,early_gb_multiplier=4,enhanced_layers=[2,6],
        unchanged_full_layers=[8,10,12],reference_loss_unchanged=True,
        preflight='Original five cached train cases; parent-equivalent full Loss/late layers, early x4 terms and aggregate delta/gradients, single disposable guarded update. Inherited continuation source proof; no formal run/dev/test/postprocessing.',
        queue='Prepared single arm: explicit --start -> train -> sole own-Loss-best freeze -> original dev500 postprocessing -> comparison with frozen existing E139; no auto-launch during preparation')
    save(HERE/'plan.json',plan)
    evalplan=json.loads((PARENT/'evaluation/plan.json').read_text())
    evalplan.update(revision='E140.frontend_gb4.v1.seed0.dev500',models={},
        authorization='Preparation only. Original dev500 evaluation is staged after a future explicit queue start.')
    save(HERE/'evaluation/plan.json',evalplan)
    save(HERE/'provenance.json',dict(parent=PARENT.relative_to(ROOT).as_posix(),
        parent_preflight_sha256=sha(PARENT/'preflight.json'),identity_mapping=MAPPING,copied_originals=copied,
        changes=['S2/S6 original weighted grouping and boundary contributions multiplied by4, overlap remains omitted',
                 'runtime identity replacements','single E139-comparison report','adapted preparation checks only'],
        parent_control=controls[0]))
    save(HERE/'authorization.json',dict(scope='E140.frontend_gb4.v1.seed0',training_enabled=False,allowed_arms=[],
        prepared_arms=[ARM],automatic_evaluation=True,test_read=False,automatic_retry=False,
        note='Prepare only. No formal training starts until future explicit user invocation of run_queue --start.'))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None,formal_training_started=False))
    from .finalize_sources import main as finalize
    finalize()
    print('Independent E140 sources prepared; training disabled',flush=True)


if __name__=='__main__':main()
