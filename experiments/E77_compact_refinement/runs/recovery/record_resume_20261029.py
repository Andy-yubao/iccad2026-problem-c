"""Record the authorized live controller without changing training identities."""
import json,re
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'experiments/E77_compact_refinement'
OUTPUT=HERE/'runs/20261029_e76_compact_corrector_seed0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=read(HERE/'state.json');assert state['controller_pid']==21280 and state['status']=='running'
calls=read(OUTPUT/'invocations.json');assert calls[-1]['resume'] and calls[-1]['prior_seconds']==4629.639423
auth=read(HERE/'runs/recovery/resume_authorization_20261029.json')
progress=read(OUTPUT/'progress.json') if (OUTPUT/'progress.json').exists() else None
logged_steps=[int(x) for x in re.findall(r'^step=(\d+) epoch=',(HERE/'runs/train.log').read_text(encoding='utf-8'),re.M)]
observed_step=max(logged_steps+[progress['step'] if progress else 0])
budget=read(OUTPUT/'budget.json')
utc=datetime.now(timezone.utc).isoformat()
path=ROOT/'experiments/E72_topology_posttrain/study.json';study=read(path)
entry=next(v for v in study['run_ledger'] if v['experiment']==HERE.name)
entry.update(status='running',controller_pid=state['controller_pid'],child_pid=state['child_pid'],
    controller_exec_session=None,launch_mode='Detached hidden Windows process; live OS PID, no unified exec session',
    launched_utc='2026-10-29T02:08:30.3868391Z',
    controller_stdout='runs/controller_resume_20261029_020830.stdout.log',
    controller_stderr='runs/controller_resume_20261029_020830.stderr.log',
    user_resume_authorization='runs/recovery/resume_authorization_20261029.json',
    actual_processes_absent_verified=False,elapsed_seconds=budget['elapsed_seconds'])
if progress:
    entry.update(last_verified_step=observed_step,last_verified_epoch=progress['epoch'],
        best_epoch=progress['best_epoch'],steps=observed_step,epochs=progress['epoch'],durable_checkpoint_step=progress['step'])
study.update(status='running',current_stage='train',controller_exec_session=None,
    next_decision='Finish original E77 within remaining original budget, then unique best dev/Q/gallery and final audit; do not start seventh candidate',
    preflight='E77 five-case physical frame, frozen source, unified Loss, CUDA and exact Adam resume passed; same sources')
study['current_user_pause'].update(resume_authorized_utc=auth['authorized_utc'],
    resume_user_message=auth['user_message'],resume_consumed=True,automation_status='ACTIVE',
    resume_requires_new_user_authorization=False)
study['last_check']={'utc':utc,'state':'live_resume_updates_verified' if observed_step else 'live_resume_loading',
    'experiment':HERE.name,'controller_pid':21280,'child_pid':state['child_pid'],
    'step':observed_step,'prior_budget_preserved':True}
study['cumulative_training_seconds']=sum(v.get('elapsed_seconds',0) for v in study['run_ledger'])
write(path,study)
print('Resume recorded; step',progress['step'] if progress else 'loading','child',state['child_pid'])
