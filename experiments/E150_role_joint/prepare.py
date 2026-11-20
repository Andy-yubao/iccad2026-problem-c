"""Independent scratch joint experiment, no historical model/source mutation."""
from pathlib import Path
import json
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
INITIAL=ROOT/'experiments/E149_role_initial'
BACKEND=ROOT/'experiments/E148_shared_functional_shape'
ARM='e150_role_joint'
SCOPE='E150.role_joint.v1.seed0'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if (HERE/'plan.json').exists():raise RuntimeError('Already prepared')
    copied={}
    names=('features.py','batches.py','ports.py','public.py','role_features.py','role_cache.py','base_data.py','base_resident.py','data.py','resident.py','augmentation.py','train.py','cuda_step.py','captured_projection.py','step_control.py','schedule.py','io.py','simple_loss.py','group_tree.py','run_queue.py')
    for name in names:
        p=INITIAL/name;raw=p.read_bytes();s=p.read_text(encoding='utf-8')
        s=s.replace('experiments.E149_role_initial','experiments.E150_role_joint').replace('E149.role_initial.v1.seed0',SCOPE)
        if name=='train.py':s=s.replace('width=64, layers=2,','width=64, layers=12,').replace('one authorized scratch36k arm','one joint scratch36k arm')
        if name=='role_cache.py':
            s=s.replace('from .io import HERE,save,sha,checkpoint','from .io import HERE,ROOT,save,sha,checkpoint')
            s=s.replace("base=HERE/'role_cache_v1'/key/dataset.role;base.mkdir(parents=True,exist_ok=True)","base=(ROOT/plan['role_cache_root'])/key/dataset.role\n    if not base.is_dir():raise RuntimeError('Registered existing role cache missing')")
            s=s.replace("if index not in content:\n                content[index]", "if index not in content:\n                raise RuntimeError('Complete existing role cache required; no rebuilding')\n                content[index]")
        if name=='run_queue.py':
            s=s.replace('initial-only training','joint training with initial supervision')
            s=s.replace('dev_initial_no_overlap_loss','dev_joint_loss').replace('train_initial_no_overlap_loss','train_joint_loss').replace('dev_initial_full_loss_diagnostic','dev_final_full_loss')
            s=s.replace("dev_final_full_loss=entry['dev_monitor']['full_total'],dev_components", "dev_final_full_loss=entry['dev_monitor']['final_total'],dev_initial_no_overlap_loss=entry['dev_monitor']['initial_no_overlap_total'],dev_initial_full_loss=entry['dev_monitor']['initial_full_total'],dev_components")
            s=s.replace("selection='initial_no_overlap dev500 arithmetic mean'","selection='joint L0/L4/L8/L12 weighted dev500 arithmetic mean'")
            s=s.replace('initial_full_diagnostic=','final_full=').replace("source=experiments/E149_role_initial/", "source=experiments/E150_role_joint/")
        if name in ('features.py','batches.py','ports.py','role_features.py'): (HERE/name).write_bytes(raw)
        else:(HERE/name).write_text(s,encoding='utf-8')
        copied[name]=dict(parent=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())
    for old,new in ((INITIAL/'model.py','initial_model.py'),(BACKEND/'model.py','refinement.py')):
        (HERE/new).write_bytes(old.read_bytes());copied[new]=dict(parent=str(old.relative_to(ROOT)),sha256=hashlib.sha256(old.read_bytes()).hexdigest())
    for name in ('e119_model.py','original_model.py','geometry_feedback.py','shape_feedback.py'):
        p=BACKEND/name;(HERE/name).write_bytes(p.read_bytes());copied[name]=dict(parent=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    raw=(BACKEND/'loss.py').read_text(encoding='utf-8');raw=raw[:raw.index('def supervised_per_case(')]
    (HERE/'base_loss.py').write_text(raw,encoding='utf-8')
    plan=json.loads((INITIAL/'plan.json').read_text(encoding='utf-8'))
    plan.update(experiment='E150 role initial plus original E148 refinement, joint scratch with initial supervision',revision=SCOPE,
        authorization='2026-11-18 user requested new initial-generator structure + original downstream structure all from scratch, initial S0 output supervised too; prepare and launch background. Continuing Loss-only scope.',
        architecture={'e150':'E149 role initial107739 + E148 public encoder101184 + three reusable two-layer units472218, original12 refinements'},
        parameters={'e150':681141},trainable_parameters={'e150':681141},
        initialization='All681141 parameters scratch seed0, no checkpoint or optimizer loaded; discarded preflight model not reused',
        supervision={'L0':1/15,'L4':1/15,'L8':1/15,'L12':.8},no_overlap_layers=[0,4],full_loss_layers=[8,12],
        loss='Original base coefficients; L0/L4 exclude all direct collision terms, L8/L12 full; auxiliary total.2 split equally across0/4/8, terminal.8',
        training_mode='One joint optimizer; initial generator, separate downstream public encoder, all three shared units trainable. Layout gradients cross all unit boundaries.',
        unit_order=[0,0,1,1,2,2],initial_readout_in_returned_states=True,
        role_cache_root='experiments/E149_role_initial/role_cache_v1',role_cache_policy='Require complete identity-compatible E149 public-role sidecars, no rebuilding or copying payloads',
        cache_policy='Reuse original shared CPU cache and complete E149 public-role sidecars; features/batches/ports/role producer byte-identical',
        terminal_evaluation='Freeze own joint-Loss best and Loss-only summary; no Q/postprocessing/test',
        queue='Joint scratch train -> frozen own joint-Loss-best -> Loss summary only',
        preflight='Copied structure identity, role cache identity, initial objective and all4 Loss gradients,13 actual readouts,hidden isolation/layout gradients,labels/permutations/augmentation,CUDA64/tail32,shared-Adam resume parity',
        reporting='Joint aggregate plus individual L0/L4/L8/L12 Loss/components, initial no-overlap/full and final complete Loss; same joint aggregate selects best/LR/earlystop',
        budget='One whole joint candidate4h/200epoch, batch64/tail32, original FP32/Adam/guards/schedule',
        comparison='E148 supervision and initial architecture differ; frozen transfer was no-training diagnostic, not a matched training baseline',
        overlap_omission='Only L0 and L4 omit entire direct overlap group; no hard prediction mask or overwrite',
        automatic_evaluation=False,automatic_retry=False,test_read=False,postprocessing_calls=0,Q_calls=0)
    save(HERE/'plan.json',plan)
    save(HERE/'authorization.json',dict(scope=SCOPE,training_enabled=False,allowed_arms=[],prepared_arms=[ARM],automatic_evaluation=False,automatic_retry=False,test_read=False,note=plan['authorization']))
    save(HERE/'queue_state.json',dict(status='prepared',completed=[],active=None))
    save(HERE/'provenance.json',dict(initial_parent=str(INITIAL.relative_to(ROOT)),refinement_parent=str(BACKEND.relative_to(ROOT)),copied=copied,
        changes=['New role initial structure + unchanged original12-layer reusable backend','All parameters joint scratch; L0 supervised too','Complete existing data/role caches reused, Loss only']))
    print('Independent E150 source and plan prepared; training still disabled.')

if __name__=='__main__':main()
