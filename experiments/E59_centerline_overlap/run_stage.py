"""Bounded serial stage execution; never select a checkpoint by terminal Q."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import HERE,ROOT,save,exclusive

CONTROL=HERE/'runs/20261027_control'
TASKS={
 'wide_repair':dict(name='20261027_wide_graph_prenorm_seed1',arm='repair160',size=16000,seed=1,hours=2.),
 'small_original':dict(name='20261027_small4096_original_seed0',arm='original96',size=4096,seed=0,hours=2.),
 'small_centerline':dict(name='20261027_small4096_centerline_seed0',arm='centerline96',size=4096,seed=0,hours=2.),
 'large_original':dict(name='20261027_large64000_original_seed0',arm='large_original160',size=64000,seed=0,hours=4.),
 'large_centerline':dict(name='20261027_large64000_centerline_seed0',arm='large_centerline160',size=64000,seed=0,hours=4.),
}


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def health(folder):
    import torch
    from .cache import load_rows
    from .data import collate
    from .model import GeometryModel
    from .batches import inputs_only
    torch.set_num_threads(1)
    state=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    ids=read(HERE/'loss_scale.json')['train_indices'];rows=load_rows('train',ids);g,_=collate(rows,'cpu')
    m=GeometryModel(**state['model_config']).eval();m.load_state_dict(state['model'])
    zero=[];free=g['valid']&g['mask'][...,0].bool()
    h=m.head[1].register_forward_hook(lambda module,args,out:zero.append(float((out[free]==0).float().mean())))
    with torch.no_grad():out=m(inputs_only(g))
    h.remove();hidden=float(m.last_hidden_max)
    result={'status':'passed' if hidden<1e5 and zero[-1]<.99 and bool(torch.isfinite(out).all()) else 'failed',
            'latest_hidden_max':hidden,'final_silu_zero_fraction':zero[-1],
            'unique_train_cases':5,'same_probe_members':ids,'test_read':False,
            'limit':'Five-case activation audit plus full training guards; not proof of universal stability'}
    save(folder/'health.json',result)
    if result['status']!='passed':raise RuntimeError('Wide model still unhealthy; inspect evidence before next stage')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',nargs='+',choices=TASKS,default=['wide_repair','small_original','small_centerline'])
    args=ap.parse_args();CONTROL.mkdir(parents=True,exist_ok=True)
    with exclusive(CONTROL/'study.lock'):
        state=read(HERE/'state.json')
        deadline=datetime.fromisoformat(state['calendar_deadline_utc']).timestamp()
        completed=state.setdefault('completed_stages',[]);started=state.setdefault('started_runs',[])
        def refresh(status,stage,child=None):
            state.update(status=status,stage=stage,controller_pid=os.getpid(),active_children=[child] if child else [],
                         updated_utc=datetime.now(timezone.utc).isoformat())
            state['new_formal_runs_started']=len(started)
            state['new_training_seconds_used']=sum(read(HERE/'runs'/name/'budget.json')['elapsed_seconds'] for name in started if (HERE/'runs'/name/'budget.json').exists())
            save(HERE/'state.json',state)
        def stopped():return (HERE/'runs/STOP').exists() or time.time()>=deadline
        def execute(task,stage,command):
            folder=HERE/'runs'/task['name'];logs=CONTROL/task['name'];logs.mkdir(exist_ok=True)
            with (logs/'console.log').open('a',encoding='utf-8') as log:
                log.write('\n'+subprocess.list2cmdline(command)+'\n');log.flush()
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                refresh('running',stage,{'pid':proc.pid,'task':task['name'],'stage':stage,'command':command,'log':str(logs/'console.log')})
                stopping=None
                while proc.poll() is None:
                    if stopped():
                        folder.mkdir(exist_ok=True);(folder/'STOP').touch();stopping=stopping or time.time()
                        if time.time()-stopping>120:
                            proc.terminate();proc.wait(timeout=30)
                            raise RuntimeError('Stage exceeded safe-stop grace; retained checkpoints')
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                if proc.returncode:raise RuntimeError(f'{stage} failed ({proc.returncode}); see {logs}')
        py=[sys.executable,'-u','-B','-m']
        try:
            if read(HERE/'runs/20261027_preflight/results.json')['status']!='passed':raise RuntimeError('Preflight required')
            for key in args.tasks:
                if key in completed:continue
                if stopped():refresh('resource_stop','calendar_or_STOP');return
                if key.startswith('large_'):
                    expected='centerline_v1' if key=='large_centerline' else 'original'
                    if state.get('selected_loss')!=expected:raise RuntimeError('Large training requires recorded paired selection')
                    if 'wide_repair' not in completed:raise RuntimeError('Wide repair not accepted')
                task=TASKS[key];folder=HERE/'runs'/task['name']
                if task['name'] not in started:
                    if len(started)>=5 or state.get('new_training_seconds_used',0)>=43200:raise RuntimeError('New-work training budget exhausted')
                    started.append(task['name']);refresh('preparing',key)
                result=read(folder/'results.json') if (folder/'results.json').exists() else None
                if result is None or result['status'] not in ('early_stop','budget','max_epochs'):
                    cmd=py+['experiments.E59_centerline_overlap.train','--output',str(folder),'--arm',task['arm'],
                            '--dataset-size',str(task['size']),'--seed',str(task['seed']),'--max-hours',str(task['hours']),
                            '--max-epochs','200','--block-steps','500','--batch-size','32','--engine','staged_cuda_graph','--deadline',str(deadline)]
                    if (folder/'checkpoints/latest.pt').exists():cmd.append('--resume')
                    execute(task,key+'_train',cmd)
                if read(folder/'results.json')['status'] not in ('early_stop','budget','max_epochs'):raise RuntimeError('Training did not finish normally')
                if key=='wide_repair':health(folder)
                for stage,extra in [('raw2000',['final','--stage','raw']),('flash500',['final','--stage','terminal']),('draft100',['render_drafts'])]:
                    if stopped():refresh('resource_stop',stage);return
                    execute(task,key+'_'+stage,py+['experiments.E59_centerline_overlap.'+extra[0],'--output',str(folder)]+extra[1:])
                if not (folder/'terminal/results.json').exists():raise RuntimeError('Missing one-best terminal evaluation')
                completed.append(key);refresh('running',key+'_complete')
            state['next_action']='Review paired small-model common metrics and 3-5 same-case images, persist provisional selected_loss using preregistered gates, then run large_original or large_centerline. Do not re-run completed stages.'
            refresh('awaiting_paired_selection' if not any(x.startswith('large_') for x in args.tasks) else 'awaiting_final_report','paired_selection' if not any(x.startswith('large_') for x in args.tasks) else 'final_report')
        except BaseException as error:
            state['last_error']=str(error);state['next_action']='Inspect failure logs/evidence and current child state; diagnose before modifying or retrying. Preserve cumulative budgets.'
            refresh('needs_diagnosis',state.get('stage'));raise


if __name__=='__main__':main()
