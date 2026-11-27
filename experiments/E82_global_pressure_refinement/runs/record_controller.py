"""Read current E81 state/progress/budget; never overwrites another phase."""
import json,re
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
study=read(ROOT/'experiments/E81_multisource_refinement/study.json');state=read(HERE/'state.json');launch=read(HERE/'runs/controller_launch.json')
output=HERE/'runs/20261029_e81_pressure_seed0'
progress=read(output/'progress.json') if (output/'progress.json').exists() else None
budget=read(output/'budget.json') if (output/'budget.json').exists() else {'elapsed_seconds':0}
result=read(output/'results.json') if (output/'results.json').exists() else None
log=(HERE/'runs/train.log').read_text(encoding='utf-8') if (HERE/'runs/train.log').exists() else ''
observed=max([int(x) for x in re.findall(r'^step=(\d+) epoch=',log,re.M)]+[progress['step'] if progress else 0])
entry={'experiment':HERE.name,'phase_candidate':2,'output':output.relative_to(HERE).as_posix(),
    'status':state['status']+'_'+state['stage'],'max_training_seconds':7200,
    'elapsed_seconds':max(budget['elapsed_seconds'],result['elapsed_seconds'] if result else 0),
    'controller_pid':state.get('controller_pid'),'child_pid':state.get('child_pid'),
    'launch':launch,'steps':observed,'durable_checkpoint_step':progress['step'] if progress else None,
    'epochs':progress['epoch'] if progress else 0,'best_epoch':progress['best_epoch'] if progress else None,
    'training_model_parameters':7545329,'deployment_parameters':1715390,'trainable_parameters':2304,
    'test_read':False,'optimizer':'Fresh AdamW only2304 pressure weights; all E81 source weights frozen, same four LR rule',
    'preflight':'runs/preflight/results.json','two_draft_views_per_member':True}
prior={r['experiment']:r for r in study['run_ledger']};prior[HERE.name]=entry;study['run_ledger']=list(prior.values())
study.update(status=state['status'],current_stage=state['stage'],current_experiment=HERE.name,
  phase_training_seconds=sum(r.get('elapsed_seconds',0) for r in study['run_ledger']))
study['global_charged_seconds']=study['previous_global_charged_seconds']+study['phase_training_seconds']
study['last_check']={'utc':datetime.now(timezone.utc).isoformat(),'observed_step':observed,
    'evidence':'State/progress/log/budget only; PID numbers alone do not prove process liveness'}
save(ROOT/'experiments/E81_multisource_refinement/study.json',study)
print(json.dumps({'status':entry['status'],'step':observed,'elapsed_seconds':entry['elapsed_seconds'],'controller_pid':entry['controller_pid'],'child_pid':entry['child_pid']}))
