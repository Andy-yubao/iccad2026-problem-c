"""Read actual E84 state/budget into the central ledger; never restart a job."""
import sys,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.E84_loss_control.io import HERE,save,exclusive
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    state=read(HERE/'state.json')
    if state.get('status')=='complete_audited':
        print('E84 is already audited and closed; do not overwrite the retired ledger.',flush=True);return
    out=Path(state['output']);central=ROOT/'experiments/E81_multisource_refinement/study.json';ledger=read(central)
    budget=read(out/'budget.json') if (out/'budget.json').exists() else {}
    progress=read(out/'progress.json') if (out/'progress.json').exists() else {}
    result=read(out/'results.json') if (out/'results.json').exists() else {}
    seconds=float(result.get('elapsed_seconds',budget.get('elapsed_seconds',progress.get('elapsed_seconds',0.))))
    entry=next((r for r in ledger['run_ledger'] if r['experiment']=='E84_loss_control'),None)
    if entry is None:
        entry={'experiment':'E84_loss_control','phase_candidate':3,'output':out.relative_to(HERE).as_posix(),
          'max_training_seconds':7200,'source_checkpoint':'Same F79 warm-wide initialization recipe as E81 epoch0, not E81 best',
          'loss':'G only per source; mean+.25*(max-mean); all original G terms','parameters':7543025,'trainable_parameters':241547,
          'test_read':False,'terminal_runs':0};ledger['run_ledger'].append(entry)
    entry.update(status=state['status'],stage=state['stage'],elapsed_seconds=seconds,
      controller_pid=state.get('controller_pid'),child_pid=state.get('child_pid'),controller_exec_session=99826,
      launch_mode='Managed persistent exec controller',updated_utc=state['updated_utc'])
    if progress:entry['progress']= {k:progress[k] for k in ('step','epoch','best_epoch','elapsed_seconds') if k in progress}
    if result:entry.update(epochs=result.get('epochs'),best_epoch=result.get('best_epoch'),stop_reason=result.get('status'))
    entry['terminal_runs']=int((out/'terminal/results.json').exists())
    ledger['phase_training_seconds']=sum(float(r.get('elapsed_seconds',0)) for r in ledger['run_ledger'])
    ledger['global_charged_seconds']=ledger['previous_global_charged_seconds']+ledger['phase_training_seconds']
    ledger.update(status='active_single_objective_control',current_experiment='E84_loss_control',current_stage=state['stage'],remaining_formal_training_slots=0,automation_status='ACTIVE')
    ledger['last_check']={'utc':datetime.now(timezone.utc).isoformat(),'state':'Actual E84 '+state['status']+'/'+state['stage']+'; prior E83 complete audited','controller_exec_session':99826}
    save(central,ledger)
    print(json.dumps({'controller':state.get('controller_pid'),'child':state.get('child_pid'),'stage':state['stage'],'accounted_training_seconds':seconds,'global_training_seconds':ledger['global_charged_seconds']}),flush=True)
if __name__=='__main__':main()
