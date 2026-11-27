"""Record latest user pause without touching production source identities."""
import json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'experiments/E77_compact_refinement'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
utc=datetime.now(timezone.utc).isoformat()
recovery=read(HERE/'runs/recovery/startup_interruption.json')
state=read(HERE/'state.json')
state['events'].append({'utc':utc,'status':'paused_by_user','stage':'user_pause'})
state.update(status='paused_by_user',stage='user_pause',updated_utc=utc,controller_pid=None,child_pid=None,
    reason='User: 现在没有训练条件',actual_E77_processes_absent_verified=True,
    previous_detached_launch_pid=41012,formal_optimizer_updates_verified=0,
    checkpoint_role='Recovered zero-update initialization only; no formal learned result',
    automation_status='PAUSED',resume_requires_new_user_authorization=True)
write(HERE/'state.json',state)
path=ROOT/'experiments/E72_topology_posttrain/study.json';study=read(path)
entry=next((v for v in study['run_ledger'] if v['experiment']==HERE.name),None)
if entry is None:
    entry={'experiment':HERE.name};study['run_ledger'].append(entry)
entry.update(output='runs/20261029_e76_compact_corrector_seed0',status='user_paused_before_formal_updates',
    dataset_size=32000,max_training_seconds=7200,parameters=5873659,trainable_parameters=21860,
    frozen_parameters=5851799,steps=0,epochs=0,source_checkpoint='E76 own-Loss best29, permanently frozen',
    loss='unchanged G+2I+5E',controller_exec_session=None,controller_pid=None,
    original_tool_session=53415,detached_launch_pid=41012,actual_processes_absent_verified=True,
    elapsed_seconds=recovery['charged_interruption_seconds'],
    budget_note='Conservatively charged unknown startup interruption; not measured active training time',
    recovery='runs/recovery/startup_interruption.json',best_epoch=None,terminal_runs=0)
study.update(status='paused_by_user',current_experiment=HERE.name,current_stage='user_pause',controller_exec_session=None,
    next_decision='Wait for explicit new training authorization; do not remove STOP or resume automatically')
study['current_user_pause']={'utc':utc,'message':'现在没有训练条件','stop_file':'experiments/E77_compact_refinement/runs/STOP',
    'automation_status':'PAUSED','actual_E77_processes_absent_verified':True,'resume_requires_new_user_authorization':True}
study['cumulative_training_seconds']=sum(x.get('elapsed_seconds',0) for x in study['run_ledger'])
study['cumulative_budget_note']='Includes conservatively charged E77 startup interruption; not all active computation.'
study['last_check']={'utc':utc,'state':'user_paused_no_active_E77_process','experiment':HERE.name,'step':0}
study['workspace_archive']['active_directories']=sorted(p.name for p in (ROOT/'experiments').iterdir() if p.is_dir() and p.name.startswith('E'))
write(path,study)
print('Recorded user pause; no formal E77 optimizer updates or evaluation.')
