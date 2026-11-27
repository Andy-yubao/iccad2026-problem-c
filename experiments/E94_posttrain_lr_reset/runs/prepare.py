"""Construct E94 fixed plan and independent evaluator before any training."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=ROOT/'experiments/E94_posttrain_lr_reset';OLD=ROOT/'experiments/E93_matched_posttrain';HOME=HERE/'qualification'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
authorization='用户：请开始一个新的训练，后训练阶段lr从1e-3开始。训练完后自动验证500例子'
plan=json.loads((OLD/'plan.json').read_text(encoding='utf-8'))
plan.update(status='authorized_training_and_once_dev500',arms=['F'],execution_order=['F'],maximum_hours_per_arm=2,
            maximum_phase_epochs=64,maximum_formal_training_seconds=7200,overlap_weights={'F':.08975162861857286},
            lr_levels=[.001,.00025,.0001,.00005],required_lr_drops=3,authorization=authorization,
            optimizer='same P best36 AdamW moments/step; only LR reset1e-3; all trainable',
            selection='same fixed F Loss for backward/train/dev/best/LR/early-stop; no Q selection',
            evaluation='automatic own-best dev500 Loss and once flash500,4process x2threads,same100 gallery; no dev2000; sealed tests',
            budget='new single candidate2h/64 epochs; preflight and <=1h evaluation separate; all old charges preserved',
            branch_initialization='same P36 model/Adam/projection and P39 RNG as E93 C; new registered schedule/counters only')
save(HERE/'plan.json',plan)
provenance=json.loads((OLD/'provenance.json').read_text(encoding='utf-8'))
provenance.update(experiment=HERE.name,authorization=authorization,independent_source=OLD.relative_to(ROOT).as_posix(),
    copied_files={p.name:sha(p) for p in OLD.iterdir() if p.is_file() and (p.suffix=='.py' or p.name=='requirements-native.txt')},
    comparison='E93 C own-best vs new F own-best; same P36 and P39 input state; different LR schedule, cap and realized exposure')
for name in ('runs/20261101_c_seed0/checkpoints/C_best.pt','runs/20261101_g_seed0/checkpoints/G_best.pt'):
    path=OLD/name;provenance['protected_artifacts'][path.relative_to(ROOT).as_posix()]=sha(path)
save(HERE/'provenance.json',provenance)
reg=json.loads((OLD/'runs/evaluation_registration.json').read_text(encoding='utf-8'))
reg.pop('candidates');reg.update(authorization=authorization,selection='freeze future unique F own-Loss best only after normal training endpoint',
    raw_count=500,evaluation_budget_seconds=3600,next_action='report only; no automatic new training/corrector/automation')
for a in ('c','g'):
    p=OLD/f'runs/20261101_{a}_seed0/qualification/terminal/results.json'
    reg['previous_results']['E93_'+a.upper()]=p.relative_to(ROOT).as_posix();reg['evidence_files'][p.relative_to(ROOT).as_posix()]=sha(p)
save(HERE/'runs/evaluation_registration.json',reg)
save(HERE/'state.json',{'status':'prepared','scope':'new F from P36 LR1e-3 then automatic once dev500','test_read':False})
save(HERE/'runs/authorization.json',{'authorization':authorization,'resolved_initial_lr':.001,'conflicting_earlier_mention':.01,
    'resolution':'explicit final number1e-3 controls','training_cap_seconds':7200,'evaluation_cap_seconds':3600,
    'historical_ledgers_read_only':['E92','E93'],'test_read':False})
# E93 raw/full-dev evaluator is deliberately narrowed to fixed500 terminal only.
run=(OLD/'qualification/run.py').read_text(encoding='utf-8')
run=run[:run.index('\ndef run():')]
run=run.replace('E93_EVAL_THREADS','E94_EVAL_THREADS').replace('Frozen F-best qualification only; resumable raw2000, one flash500 and gate.','E94 unique F-best fixed500 evaluation; no full-dev or training.')
run+='''
def run():
    begin=time.perf_counter();prior=auxiliary_seconds();limit=registration()['evaluation_budget_seconds'];last=[0.]
    def halt():return (OUTPUT/'STOP').exists() or (HERE/'runs/STOP').exists() or prior+time.perf_counter()-begin>=limit
    def beat(activity,**fields):
        now=time.perf_counter()-begin
        if now-last[0]>=5. or activity!=getattr(beat,'activity',None):
            save(OUTPUT/'heartbeat.json',{'activity':activity,**fields,'elapsed_this_invocation':now,'remaining_seconds':max(0.,limit-prior-now),'test_read':False})
            beat.activity=activity;last[0]=now
    try:
        with exclusive(HERE/'runs/training.lock'),exclusive(OUTPUT.with_suffix('.lock')):
            state,identity=freeze();snap=OUTPUT/'source_snapshot.zip'
            if not snap.exists():
                with zipfile.ZipFile(snap,'w',zipfile.ZIP_DEFLATED) as archive:
                    for name in identity['sources']:archive.write(ROOT/name,name)
            if halt():raise EvaluationStop()
            monitor500(state,identity,halt,beat);del state;torch.cuda.empty_cache()
            terminal(identity,halt,beat)
            if halt():raise EvaluationStop()
            from .render import run as render
            beat('same100_gallery');render(OUTPUT,halt)
            from .compare import assess
            beat('saved_result_comparison');print(json.dumps(assess(identity),ensure_ascii=False,indent=2),flush=True)
            save(OUTPUT/'status.json',{'status':'completed','test_read':False});beat('completed')
    except EvaluationStop:save(OUTPUT/'status.json',{'status':'stopped','test_read':False})
    except BaseException as error:
        save(OUTPUT/'status.json',{'status':'error','error':repr(error),'test_read':False});raise
    finally:charge_auxiliary(time.perf_counter()-begin,'frozen E94 once500')

def monitor500(state,identity,halt,beat):
    target=OUTPUT/'dev500_loss.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))['checkpoint_sha256']!=identity['checkpoint_sha256']:raise ValueError('Loss model drift')
        return
    seed_all(0,4);model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    rows=load_rows('dev_monitor500');pack=ResidentRows(rows,'cuda',max_nodes=120);register_resident(pack)
    beat('dev500_own_loss');value=training_monitor(model,rows,'F','cuda',halt)
    if value is None:raise EvaluationStop()
    if abs(value['total']-identity['selected_dev500_loss'])>1e-5:raise ValueError('Best monitor mismatch')
    save(target,{'checkpoint_sha256':identity['checkpoint_sha256'],'losses':value,'selected_training_monitor':identity['selected_dev500_loss'],'test_read':False})

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old=signal.signal(signal.SIGINT,lambda *_:(OUTPUT/'STOP').touch())
    try:run()
    finally:signal.signal(signal.SIGINT,old)
'''
# Remove the unused full-dev inference function to make the500-only boundary explicit.
start=run.index('\ndef raw_dev(');end=run.index('\ndef validate_terminal(')
run=run[:start]+run[end:]
(HOME/'run.py').write_text(run,encoding='utf-8')
for name in ('__init__.py','render.py'):
    p=HOME/name;p.write_text(p.read_text(encoding='utf-8').replace('E93','E94'),encoding='utf-8')
print('E94 fixed plan/evaluation registration prepared')
