"""One-time construction of independent E93 evaluator from frozen E92 files."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'experiments/E93_matched_posttrain';OLD=ROOT/'experiments/E92_two_stage_placer/qualification';HOME=HERE/'qualification'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

run=(OLD/'run.py').read_text(encoding='utf-8').split('\ndef gate(')[0]
run=run.replace('from .common import OUTPUT,REGISTRATION,registration,ready,freeze,auxiliary_seconds,charge_auxiliary,seed_all','from .common import ARM,RUN,OUTPUT,REGISTRATION,registration,ready,freeze,auxiliary_seconds,charge_auxiliary,seed_all')
run=run.replace("'F'","ARM").replace('selected_dev500_F_loss','selected_dev500_loss').replace('E92_EVAL_THREADS','E93_EVAL_THREADS').replace('dev500_F_loss','dev500_own_loss')
run+='''
def run():
    begin=time.perf_counter();prior=auxiliary_seconds();limit=registration()['evaluation_budget_seconds'];last=[0.]
    def halt():return (OUTPUT/'STOP').exists() or (HERE/'runs/evaluation/STOP').exists() or prior+time.perf_counter()-begin>=limit
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
            raw_dev(state,identity,halt,beat);del state;torch.cuda.empty_cache()
            terminal(identity,halt,beat)
            if halt():raise EvaluationStop()
            from .render import run as render
            beat('same100_gallery');render(OUTPUT,halt)
            from .compare import assess
            beat('saved_result_comparison');print(json.dumps(assess(identity),ensure_ascii=False,indent=2),flush=True)
            save(OUTPUT/'status.json',{'status':'completed','test_read':False})
            beat('completed')
    except EvaluationStop:
        save(OUTPUT/'status.json',{'status':'stopped','test_read':False})
    except BaseException as error:
        save(OUTPUT/'status.json',{'status':'error','error':repr(error),'test_read':False});raise
    finally:charge_auxiliary(time.perf_counter()-begin,'frozen evaluation '+ARM)

if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old=signal.signal(signal.SIGINT,lambda *_:(OUTPUT/'STOP').touch())
    try:run()
    finally:signal.signal(signal.SIGINT,old)
'''
(HOME/'run.py').write_text(run,encoding='utf-8')
for name in ('__init__.py','render.py'):
    p=HOME/name;value=p.read_text(encoding='utf-8').replace('E92_EVAL_THREADS','E93_EVAL_THREADS')
    if name=='render.py':value=value.replace('E92 frozen F own-best','E93 frozen own-best').replace('E92 F-best','E93 own-best').replace('E92 F own-best','E93 own-best').replace('E92','E93')
    p.write_text(value,encoding='utf-8')
import torch
oldreg=json.loads((OLD.parent/'runs/qualification_registration.json').read_text(encoding='utf-8'))
previous={'E90':'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/terminal/results.json',
          'E92':'experiments/E92_two_stage_placer/runs/20261031_two_stage32k_seed0/qualification/terminal/results.json'}
evidence=dict(oldreg['baseline']['source_files']);evidence.update({p:sha(ROOT/p) for p in previous.values()})
candidates={}
for arm in ('C','G'):
    folder=HERE/f'runs/20261101_{arm.lower()}_seed0';path=folder/f'checkpoints/{arm}_best.pt';results=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    candidates[arm]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'best_epoch':results['best_epoch'][arm],'selected_loss':results['best'][arm]}
save(HERE/'runs/evaluation_registration.json',{'authorization':'用户2026-11-01：运行验证','selection':'unique own-arm fixed Loss best; frozen before new inference/Q',
    'candidates':candidates,'baseline':oldreg['baseline'],'previous_results':previous,'evidence_files':evidence,
    'workers':4,'threads_per_worker':2,'raw_count':2000,'terminal_count':500,'terminal_runs_per_candidate':1,
    'evaluation_budget_seconds':7200,'preparation_seconds_separate':True,'test_read':False,
    'next_action':'evaluation/report only; no Q selection, training, corrector or automation'})
manifest={p.relative_to(HERE).as_posix():{'source':p.relative_to(HERE).as_posix().replace('qualification','experiments/E92_two_stage_placer/qualification',1),'sha256':sha(p)} for p in HOME.rglob('*.py') if (OLD/p.relative_to(HOME)).exists() and p.name not in ('common.py','run.py','render.py','__init__.py')}
save(HERE/'runs/evaluation/copied_files.json',manifest)
print('Independent evaluator and fixed candidate registration prepared')
