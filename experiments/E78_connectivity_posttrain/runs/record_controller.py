"""Phase2 bookkeeping from actual pipeline/progress/logs; no training mutation."""
import json,re
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
state=read(HERE/'state.json');study=read(HERE/'study.json');entry=study['run_ledger'][0]
output=HERE/entry['output'];progress=read(output/'progress.json') if (output/'progress.json').exists() else None
budget=read(output/'budget.json') if (output/'budget.json').exists() else {'elapsed_seconds':0}
log=(HERE/'runs/train.log').read_text(encoding='utf-8')
observed=max([int(v) for v in re.findall(r'^step=(\d+) epoch=',log,re.M)]+[progress['step'] if progress else 0])
entry.update(status=state['status']+'_'+state['stage'],controller_pid=state.get('controller_pid'),
    child_pid=state.get('child_pid'),controller_exec_session=None,
    launch_mode='Detached hidden Windows process',launched_utc='2026-10-29T05:47:15.7978474Z',
    stdout='runs/controller_20261029_054715.stdout.log',stderr='runs/controller_20261029_054715.stderr.log',
    elapsed_seconds=budget['elapsed_seconds'],last_verified_step=observed,steps=observed)
if progress:entry.update(epochs=progress['epoch'],best_epoch=progress['best_epoch'],durable_checkpoint_step=progress['step'])
study.update(status=state['status'],current_stage=state['stage'],controller_pid=state.get('controller_pid'),
    child_pid=state.get('child_pid'),automation_id='problem-c',automation_status='ACTIVE',
    phase_training_seconds=sum(v.get('elapsed_seconds',0) for v in study['run_ledger']))
study['global_charged_seconds']=study['previous_charged_seconds']+study['phase_training_seconds']
study['last_check']={'utc':datetime.now(timezone.utc).isoformat(),'step':observed,
    'state':'updates_verified' if observed else 'started_loading','source_identity_frozen':True}
(HERE/'study.json').write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('E78',state['status'],state['stage'],'controller',state.get('controller_pid'),'child',state.get('child_pid'),'step',observed,flush=True)
