"""One authorized initial-only training worker, then freeze and report Loss."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import datetime
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from .io import HERE,ROOT,save,sha,sources,exclusive
from .loss_config import ARMS

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
    gate=read(HERE/'preflight.json')
    if gate['status']!='passed' or gate['sources']!=sources() or not gate['ctrl_c_resume_verified']:raise RuntimeError('Preflight/source/resume gate mismatch')
    return gate

def freeze():
    import torch
    arm=ARMS[0];run=HERE/'runs'/f'{arm}_seed0';result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result.get('error'):raise RuntimeError('Unsuccessful training; no next stage')
    path=run/'checkpoints/best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['identity']['sources']==sources()
    p=state['progress'];entry=next(r for r in p['history'] if r['epoch']==p['best_epoch'])
    summary=dict(status='complete',arm=arm,best_epoch=p['best_epoch'],checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),
        dev_initial_no_overlap_loss=entry['dev_monitor']['total'],train_initial_no_overlap_loss=entry['train_monitor']['total'],
        dev_initial_full_loss_diagnostic=entry['dev_monitor']['full_total'],dev_components=entry['dev_monitor'],
        train_components=entry['train_monitor'],training=result,parameters=sum(v.numel() for v in state['model'].values()),
        test_read=False,postprocessing_calls=0,Q_calls=0)
    save(HERE/'frozen_best.json',dict(checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256'],best_epoch=p['best_epoch'],selection='initial_no_overlap dev500 arithmetic mean',own_selection_loss=p['best']))
    save(HERE/'loss_results.json',summary)
    line=f'[LOSS] arm={arm} best_epoch={p["best_epoch"]} train={summary["train_initial_no_overlap_loss"]:.9f} dev={summary["dev_initial_no_overlap_loss"]:.9f} initial_full_diagnostic={summary["dev_initial_full_loss_diagnostic"]:.9f} source=experiments/E149_role_initial/loss_results.json'
    (HERE/'background').mkdir(exist_ok=True)
    with (HERE/'background/loss.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)

def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start',action='store_true');mode.add_argument('--resume',action='store_true');args=parser.parse_args()
    requested=[False];signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,lambda *_:requested.__setitem__(0,True))
    with exclusive(HERE/'queue.lock'):
        verify();state=read(HERE/'queue_state.json')
        if any((HERE/name).exists() for name in ('STOP','QUEUE_STOP')):raise RuntimeError('STOP present')
        if args.start:
            if state['status']!='prepared' or (HERE/'runs').exists():raise RuntimeError('Not a fresh queue')
            auth=read(HERE/'authorization.json');assert not auth['training_enabled']
            auth.update(training_enabled=True,allowed_arms=list(ARMS),activated_at=now(),preflight_sha256=sha(HERE/'preflight.json'))
            save(HERE/'authorization.json',auth)
            state=dict(status='running',pid=os.getpid(),queue_id=uuid.uuid4().hex,started_at=now(),completed=[],active=None,sessions=1)
        else:
            if state['status']!='paused':raise RuntimeError('Only safe paused queue can resume')
            marker=HERE/'PAUSE_REQUEST'
            if marker.exists():
                if read(marker)['queue_id']!=state['queue_id']:raise ValueError('Foreign pause marker')
                history=HERE/'pause_history';history.mkdir(exist_ok=True);marker.rename(history/(uuid.uuid4().hex+'.json'))
            state.update(status='running',pid=os.getpid(),sessions=state['sessions']+1)
        def beat(**kw):state.update(kw);save(HERE/'queue_state.json',dict(state,updated_at=now()))
        def pause():
            if requested[0] or any((HERE/name).exists() for name in ('STOP','QUEUE_STOP')):
                if not (HERE/'PAUSE_REQUEST').exists():save(HERE/'PAUSE_REQUEST',dict(queue_id=state['queue_id'],requested_at=now()))
                return True
            return False
        child=None
        try:
            beat()
            if 'train' not in state['completed']:
                run=HERE/'runs'/f'{ARMS[0]}_seed0'
                command=[sys.executable,'-u','-B','-m','experiments.E149_role_initial.train','--arm',ARMS[0]]
                if args.resume:command.append('--resume')
                flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
                logs=HERE/'launch_logs';logs.mkdir(exist_ok=True);log=logs/f'train_session{state["sessions"]:03d}.log'
                with log.open('x',encoding='utf-8') as sink:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
                    def pump():
                        for line in child.stdout:sink.write(line);sink.flush();print(line,end='',flush=True)
                    thread=threading.Thread(target=pump,daemon=True);thread.start()
                    beat(active=dict(phase='train',pid=child.pid,log=str(log.relative_to(ROOT))))
                    while child.poll() is None:pause();beat();time.sleep(.5)
                    thread.join()
                code=child.returncode;child=None;result=read(run/'results.json')
                if code or result['status']=='error':raise RuntimeError('Training failed; no automatic retry')
                if result['status']=='user_stop':beat(status='paused',active=None);return
                if result['status'] not in ('early_stop','budget','max_epochs'):raise RuntimeError('Unexpected terminal status')
                state['completed'].append('train');beat(active=None)
            if pause():beat(status='paused',active=None);return
            freeze();state['completed'].append('freeze_loss_summary');beat(status='complete',active=None,completed_at=now())
        except BaseException as exc:
            if child is not None and child.poll() is None:
                save(HERE/'PAUSE_REQUEST',dict(queue_id=state['queue_id'],requested_at=now()));child.wait()
            beat(status='needs_attention',active=None,error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
