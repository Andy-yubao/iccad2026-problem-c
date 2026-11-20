"""Finite user-authorized A -> B -> progressive stages -> three dev500 jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
OWNER=Path(__file__).resolve().parent
NAMES=('E109_deep_supervision_shared','E110_deep_supervision_heads','E111_progressive_depth')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def now(): return datetime.now(timezone.utc).isoformat()


def main():
    if (OWNER/'queue_state.json').exists(): raise FileExistsError('Queue already dispatched; inspect state, no automatic restart')
    jobs=[dict(name=NAMES[0],kind='train',run='20261107_shared_seed0'),
          dict(name=NAMES[1],kind='train',run='20261107_heads_seed0')]
    jobs += [dict(name=NAMES[2],kind='train',run=f'20261107_depth{depth:02d}_seed0',depth=depth) for depth in (4,8,12)]
    jobs += [dict(name=name,kind='evaluation',run=run) for name,run in
             zip(NAMES,('20261107_shared_seed0','20261107_heads_seed0','20261107_depth12_seed0'))]
    state=dict(status='running',started_at=now(),pid=os.getpid(),jobs=jobs,completed=[],active=None,
               scope='Exactly three trained models, two comparison runs plus three progressive stages, then fixed dev500 each; no test or control retraining')
    def beat(**kw):
        state.update(kw);state['updated_at']=now();save(OWNER/'queue_state.json',state)
    try:
        for index,job in enumerate(jobs):
            if (OWNER/'QUEUE_STOP').exists(): raise InterruptedError('Queue STOP')
            folder=ROOT/'experiments'/job['name'];run=folder/'runs'/job['run']
            if (folder/'STOP').exists():raise InterruptedError('Candidate STOP')
            if job.get('depth',4)>4:
                previous=folder/'runs'/f"20261107_depth{job['depth']-4:02d}_seed0/results.json"
                if read(previous)['status']!='early_stop':
                    raise RuntimeError('Intermediate progressive stage did not early-stop; no next stage or final evaluation')
                parent=folder/'runs'/f"20261107_depth{job['depth']-4:02d}_seed0/checkpoints/best.pt"
                save(folder/f"stage_handoff_depth{job['depth']:02d}.json",dict(parent_best=str(parent.relative_to(ROOT)),
                    parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),parent_result=read(previous),
                    transition='freeze old weights; new four plus shared head; inherit full best Adam/RNG; restart original LR and scheduler; charge all previous actual time'))
            if job['kind']=='train':
                result_path=run/'results.json';module=f"experiments.{job['name']}.train"
            else:
                result_path=run/'evaluation_depth_dev500/results.json';module=f"experiments.{job['name']}.evaluation.run"
            if result_path.exists():raise FileExistsError('Existing result requires identity review; no duplicate dispatch')
            env=dict(os.environ)
            if 'depth' in job:env['E111_DEPTH']=str(job['depth'])
            logs=folder/'launch_logs';logs.mkdir(exist_ok=True)
            tag=job['run']+'_'+job['kind']
            with (logs/(tag+'.stdout.log')).open('x',encoding='utf-8') as stdout, (logs/(tag+'.stderr.log')).open('x',encoding='utf-8') as stderr:
                child=subprocess.Popen([sys.executable,'-u','-B','-m',module],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                beat(active=dict(**job,pid=child.pid,index=index+1,total=len(jobs)),active_started_at=now())
                while child.poll() is None:
                    if (OWNER/'QUEUE_STOP').exists():
                        (folder/'STOP').write_text('User queue STOP; save and do not continue\n',encoding='utf-8')
                    time.sleep(5)
                if child.returncode!=0: raise RuntimeError(f'{tag} exited {child.returncode}; no automatic retry or next job')
            result=read(result_path)
            accepted=('early_stop','budget','max_epochs') if job['kind']=='train' else ('complete',)
            if result['status'] not in accepted or result.get('error'):
                raise RuntimeError(f"{tag} ended {result['status']}; no next job")
            if job['name']==NAMES[2] and job['kind']=='train':
                progress=read(run/'progress.json')
                best=next(row for row in progress['history'] if row['epoch']==result['best_epoch'])
                last=progress['history'][-1]
                save(run/'stage_summary.json',dict(active_depth=job['depth'],status=result['status'],
                    best_epoch=result['best_epoch'],best_terminal_dev_loss=best['dev_monitor']['total'],
                    corresponding_terminal_train_loss=best['train_monitor']['total'],
                    last_terminal_dev_loss=last['dev_monitor']['total'],last_terminal_train_loss=last['train_monitor']['total'],
                    accepted_updates=result['steps'],elapsed_seconds=result['elapsed_seconds'],
                    cumulative_elapsed_seconds=result['cumulative_elapsed_seconds'],
                    objective='Original E98 Loss at this stage terminal layer only; no layerwise supervision'))
            state['completed'].append(dict(**job,result=str(result_path.relative_to(ROOT)),status=result['status']))
            beat(active=None)
        evaluations={name:read(ROOT/'experiments'/name/'runs'/run/'evaluation_depth_dev500/results.json')
                     for name,run in zip(NAMES,('20261107_shared_seed0','20261107_heads_seed0','20261107_depth12_seed0'))}
        control=ROOT/'experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/evaluation_depth_dev500/results.json'
        save(OWNER/'comparison.json',dict(status='complete',models=evaluations,historical_control=read(control),
            historical_control_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),control_retrained=False,
            caveat='Single-seed historical control; loss-specific LR/early-stop and B extra parameters, progressive extra stage budget/frozen weights; timing scopes disclosed, no test inference'))
        beat(status='complete',active=None,completed_at=now())
    except BaseException as error:
        import traceback
        beat(status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
