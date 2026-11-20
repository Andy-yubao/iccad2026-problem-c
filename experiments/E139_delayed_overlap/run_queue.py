"""Foreground single-arm queue. Ctrl+C pauses; --resume continues accepted state."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONUNBUFFERED','1')
import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from .io import ROOT,HERE as OWNER,save,sha,sources,exclusive
from .loss_config import ARMS

MODULE='experiments.E139_delayed_overlap'
SCOPE='E139.delayed_overlap.v1.seed0'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def verify():
    gate=read(OWNER/'preflight.json')
    if gate['status']!='passed' or gate['sources']!=sources():raise RuntimeError('Preflight/source identity mismatch')
    if not gate['ctrl_c_resume_verified']:raise RuntimeError('Checkpoint/resume checks have not passed')
    return gate

def blocked():
    return any((OWNER/name).exists() for name in ('STOP','QUEUE_STOP'))

def freeze(arm):
    import torch
    run=OWNER/'runs'/f'{arm}_seed0'
    result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result.get('error'):
        raise RuntimeError('Training is not terminal-successful; evaluation blocked')
    checkpoint=run/'checkpoints/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    p=state['progress']
    entry=next(r for r in p['history'] if r['epoch']==p['best_epoch'])
    if p['best']!=result['best_fixed_loss']:raise ValueError('Best selection mismatch')
    if state['identity']['sources']!=sources():raise ValueError('Sources changed since training')
    value=dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(checkpoint),
        arm=arm,best_epoch=p['best_epoch'],own_selection_loss=p['best'],
        common_final_dev_loss=entry['dev_monitor']['original_final_total'],
        common_final_train_loss=entry['train_monitor']['original_final_total'],
        own_new_final_dev_loss=entry['dev_monitor']['final_total'],
        training_seconds=result['elapsed_seconds'],training_epochs=result['epochs'],training_steps=result['steps'],
        parameters=sum(v.numel() for v in state['model'].values()),frozen_before_Q=True)
    path=OWNER/'evaluation'/f'frozen_{arm}.json'
    if path.exists():
        if read(path)!=value:raise ValueError('Existing frozen model identity differs')
    else:save(path,value)

def write_comparison():
    rows=[]
    for arm in ARMS:
        result=read(OWNER/'runs'/f'{arm}_seed0/evaluation_no_mib_dev500/results.json')
        if result['status']!='complete' or result['count']!=500:raise ValueError('Evaluation incomplete')
        rows.append(dict(result,arm=arm))
    save(OWNER/'comparison.json',dict(cells=rows,historical_E119_Q_not_a_reliable_reference=True,
         note='Within-run single-seed development comparison; different own aggregate Losses are not ranked directly.'))
    text='# E139 延后重叠监督结果\n\n复用E134对照另行比较；不以历史E119 Q≈1.02作为可靠标杆。单seed、重复使用的dev和自适应早停仍有限制。\n\n'
    text+='| 组 | 混合Q↓ | 最终硬合法 | 均时/P95 ms | 共同末层新Loss |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        text+=f"| {r['arm']} | {r['mixed_Q']:.6f} | {r['hard_legal']}/500 | {r['mean_complete_seconds']*1000:.1f}/{r['p95_complete_seconds']*1000:.1f} | {r['common_final_new_dev_loss']:.6f} |\n"
    (OWNER/'comparison.md').write_text(text,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start',action='store_true')
    mode.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    requested=[False]
    signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,lambda *_:requested.__setitem__(0,True))
    with exclusive(OWNER/'queue.lock'):
        verify()
        if blocked():raise RuntimeError('STOP/QUEUE_STOP present; not automatically cleared')
        state=read(OWNER/'queue_state.json')
        if args.start:
            if state['status']!='prepared' or (OWNER/'runs').exists():raise ValueError('Not a fresh prepared queue; use --resume after Ctrl+C')
            state=dict(status='running',queue_id=uuid.uuid4().hex,started_at=now(),completed=[],active=None,sessions=0)
            auth=read(OWNER/'authorization.json')
            if auth['training_enabled']:raise ValueError('Already activated')
            auth.update(training_enabled=True,allowed_arms=list(ARMS),activated_at=now(),
                        preflight_sha256=sha(OWNER/'preflight.json'),activation='Explicit user invocation of run_queue --start')
            save(OWNER/'authorization.json',auth)
        else:
            if state['status']!='paused':raise ValueError('Resume only accepts a safely paused queue; errors/crashes require inspection')
            auth=read(OWNER/'authorization.json')
            if not auth['training_enabled'] or auth['preflight_sha256']!=sha(OWNER/'preflight.json'):raise ValueError('Activation receipt mismatch')
            marker=OWNER/'PAUSE_REQUEST'
            if marker.exists():
                if read(marker)['queue_id']!=state['queue_id']:raise ValueError('Pause marker is not from this queue')
                destination=OWNER/'pause_history'/f'{uuid.uuid4().hex}.json';destination.parent.mkdir(exist_ok=True)
                marker.rename(destination)
            state['status']='running'
        state['sessions']+=1
        state['pid']=os.getpid()
        session=f'{state["sessions"]:03d}_{uuid.uuid4().hex[:8]}'
        logs=OWNER/'launch_logs'/session;logs.mkdir(parents=True)
        child=None
        def beat(**kw):
            state.update(kw);save(OWNER/'queue_state.json',dict(state,updated_at=now()))
        def pause_request():
            if requested[0] or blocked():
                if not (OWNER/'PAUSE_REQUEST').exists():save(OWNER/'PAUSE_REQUEST',dict(queue_id=state['queue_id'],requested_at=now(),reason='Ctrl+C or STOP'))
                return True
            return False
        def launch(arm,phase):
            nonlocal child
            verify()
            run=OWNER/'runs'/f'{arm}_seed0'
            command=[sys.executable,'-u','-B','-m',MODULE+('.train' if phase=='train' else '.evaluation.run'),'--arm',arm]
            if phase=='train' and (run/'checkpoints/latest.pt').exists():command.append('--resume')
            if phase=='evaluation' and (run/'evaluation_no_mib_dev500').exists():command.append('--resume')
            flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
            logfile=logs/f'{arm}.{phase}.log'
            with logfile.open('x',encoding='utf-8') as sink:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
                def pump():
                    for line in child.stdout:
                        sink.write(line);sink.flush();print(line,end='',flush=True)
                thread=threading.Thread(target=pump,daemon=True);thread.start()
                beat(active=dict(arm=arm,phase=phase,pid=child.pid,log=logfile.relative_to(ROOT).as_posix()))
                while child.poll() is None:
                    pause_request();beat();time.sleep(.25)
                thread.join()
            code=child.returncode;child=None
            result=read(run/('results.json' if phase=='train' else 'evaluation_no_mib_dev500/state.json'))
            if result['status'] in ('error','failed') or code!=0:raise RuntimeError(f'{arm} {phase} failed; no automatic retry')
            if result['status'] in ('user_stop','paused'):return False
            expected=('early_stop','budget','max_epochs') if phase=='train' else ('complete',)
            if result['status'] not in expected:raise ValueError('Unexpected worker terminal status')
            return True
        try:
            beat()
            for arm in ARMS:
                for phase in ('train','freeze','evaluation'):
                    key=f'{arm}:{phase}'
                    if key in state['completed']:continue
                    if pause_request():
                        beat(status='paused',active=None);print('Paused safely. Continue with: python -B -m '+MODULE+'.run_queue --resume',flush=True);return
                    if phase=='freeze':freeze(arm);complete=True
                    else:complete=launch(arm,phase)
                    if complete:state['completed'].append(key)
                    beat(active=None)
                    if not complete or pause_request():
                        beat(status='paused',active=None);print('Paused safely. Continue with: python -B -m '+MODULE+'.run_queue --resume',flush=True);return
            write_comparison();beat(status='complete',active=None,completed_at=now())
            print('E139 training and original dev500 postprocessing completed.',flush=True)
        except BaseException as error:
            if child is not None and child.poll() is None:
                save(OWNER/'PAUSE_REQUEST',dict(queue_id=state['queue_id'],requested_at=now(),reason='Controller error; stop safely'))
                child.wait()
            beat(status='needs_attention',active=None,error=repr(error),traceback=traceback.format_exc())
            raise

if __name__=='__main__':main()
