"""Bounded, predefined serial capacity/data-size study. No adaptive new experiments."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from .io import ROOT, HERE, save, exclusive, sha
from .protection import verify as verify_protected

CONTROL = HERE/'runs/20261026_study_control'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE/'study_plan.json').read_text(encoding='utf-8'))
    if args.dry_run:
        print(json.dumps({'tasks':plan['tasks'],'max_hours':plan['max_hours'],'test_read':False},indent=2));return
    CONTROL.mkdir(parents=True,exist_ok=True)
    with exclusive(CONTROL/'study.lock'):
        verify_protected()
        budget_path = CONTROL/'budget.json'
        if budget_path.exists():
            budget = json.loads(budget_path.read_text(encoding='utf-8'))
            if budget['plan_sha256'] != sha(HERE/'study_plan.json'):
                raise ValueError('Cannot alter the declared study while resuming')
        else:
            now = time.time()
            budget = {'started_unix':now,'deadline_unix':now+plan['max_hours']*3600,
                      'plan_sha256':sha(HERE/'study_plan.json')}
            save(budget_path,budget)
        deadline = budget['deadline_unix']
        state_path = CONTROL/'progress.json'
        previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        completed = previous.get('completed',[]); failures = previous.get('failures',[])
        def update(status, task=None, stage=None, child_pid=None):
            save(state_path,{'status':status,'current_task':task,'stage':stage,'child_pid':child_pid,
                'completed':completed,'failures':failures,'deadline_unix':deadline,
                'elapsed_seconds':time.time()-budget['started_unix'],
                'updated_utc':datetime.now(timezone.utc).isoformat(),'test_read':False})
        def stopped():
            if time.time()>=deadline:
                (HERE/'runs/STOP').touch()
            return (HERE/'runs/STOP').exists()
        def execute(task, stage, command):
            folder = HERE/'runs'/task; log_dir = HERE/'runs'/(task+'_control');log_dir.mkdir(exist_ok=True)
            print(f'STAGE {task} {stage}: '+subprocess.list2cmdline(command),flush=True)
            with (log_dir/'console.log').open('a',encoding='utf-8') as log:
                log.write('\n'+stage+': '+subprocess.list2cmdline(command)+'\n');log.flush()
                proc = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update('running',task,stage,proc.pid)
                stop_begin = None
                while proc.poll() is None:
                    if stopped():
                        if folder.exists():(folder/'STOP').touch()
                        stop_begin = stop_begin or time.time()
                        if time.time()-stop_begin>120:
                            proc.terminate();proc.wait(timeout=30)
                            raise RuntimeError('Stage exceeded stop grace; saved outputs retained')
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                if proc.returncode:
                    raise RuntimeError(f'{stage} exited {proc.returncode}; see {log_dir}')
        py = [sys.executable,'-u','-B','-m']
        try:
            if not (HERE/'runs/symmetry_counterexample/results.json').exists() and not stopped():
                execute('symmetry_counterexample','counterexample',py+['experiments.E57_capacity_data_diagnosis.symmetry'])
            for task in plan['tasks']:
                name = task['name'];folder = HERE/'runs'/name
                if name in completed or any(x['task']==name for x in failures):continue
                if stopped():break
                try:
                    result_path = folder/'results.json'
                    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
                    if result is None or result['status'] not in ('early_stop','budget','max_epochs'):
                        cmd = py+['experiments.E57_capacity_data_diagnosis.train','--output',str(folder),
                            '--arm',task['arm'],'--seed',str(task['seed']),'--dataset-size',str(task['dataset_size']),
                            '--block-steps',str(task['block_steps']),'--max-hours',str(task['max_hours']),
                            '--max-epochs','200','--batch-size','32','--engine','cuda_graph',
                            '--deadline',str(deadline)]
                        if (folder/'checkpoints/latest.pt').exists():cmd.append('--resume')
                        execute(name,'train',cmd)
                    if stopped():break
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                    if result['status'] not in ('early_stop','budget','max_epochs'):
                        update('stopped',name,'training_not_finished');break
                    if task['kind'] != 'memory':
                        for stage,extra in [('raw2000',['final','--stage','raw']),('flash500',['final','--stage','terminal']),('draft100',['render_drafts'])]:
                            if stopped():break
                            execute(name,stage,py+['experiments.E57_capacity_data_diagnosis.'+extra[0],
                                    '--output',str(folder)]+extra[1:])
                        if stopped():break
                        if not (folder/'terminal/results.json').exists():
                            raise RuntimeError('Incomplete flash500; cannot call task complete')
                    completed.append(name);update('running',name,'complete')
                except Exception as error:
                    if stopped():break
                    failures.append({'task':name,'error':str(error)});update('running',name,'failed')
                    print('TASK FAILED; preserve evidence and continue predefined independent tasks:',name,str(error),flush=True)
                execute('study_report','interim_report',py+['experiments.E57_capacity_data_diagnosis.report'])
            if not stopped():
                execute('study_report','final_report',py+['experiments.E57_capacity_data_diagnosis.report'])
            update('stopped' if stopped() else ('complete_with_failures' if failures else 'complete'))
        except BaseException as error:
            update('failed',stage=str(error));raise
        finally:
            from .report import main as write_report
            write_report()
            verify_protected()


if __name__ == '__main__':
    main()
