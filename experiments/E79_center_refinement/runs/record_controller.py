"""Update central phase2 ledger from E79's actual state/progress/logs."""
import json,re
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
LEDGER=ROOT/'experiments/E78_connectivity_posttrain/study.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
state=read(HERE/'state.json');study=read(LEDGER)
entry=next(r for r in study['run_ledger'] if r['experiment']==HERE.name)
output=HERE/entry['output'];progress=read(output/'progress.json') if (output/'progress.json').exists() else None
budget=read(output/'budget.json') if (output/'budget.json').exists() else {'elapsed_seconds':0}
log=(HERE/'runs/train.log').read_text(encoding='utf-8')
observed=max([int(x) for x in re.findall(r'^step=(\d+) epoch=',log,re.M)]+[progress['step'] if progress else 0])
entry.update(status=state['status']+'_'+state['stage'],controller_pid=state.get('controller_pid'),child_pid=state.get('child_pid'),
    controller_exec_session=None,launch_mode='Detached hidden Windows process',launched_utc='2026-10-29T06:55:25.3340170Z',
    stdout='runs/controller_20261029_065525.stdout.log',stderr='runs/controller_20261029_065525.stderr.log',
    elapsed_seconds=budget['elapsed_seconds'],steps=observed,last_verified_step=observed)
if progress:entry.update(epochs=progress['epoch'],best_epoch=progress['best_epoch'],durable_checkpoint_step=progress['step'])
study.update(status=state['status'],current_experiment=HERE.name,current_stage=state['stage'],
    controller_pid=state.get('controller_pid'),child_pid=state.get('child_pid'),automation_status='ACTIVE',
    phase_training_seconds=sum(r.get('elapsed_seconds',0) for r in study['run_ledger']))
study['global_charged_seconds']=study['previous_charged_seconds']+study['phase_training_seconds']
study['last_check']={'utc':datetime.now(timezone.utc).isoformat(),'step':observed,'state':'updates_verified' if observed else 'started_loading'}
LEDGER.write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('E79',state['status'],state['stage'],'controller',state.get('controller_pid'),'child',state.get('child_pid'),'step',observed,flush=True)
