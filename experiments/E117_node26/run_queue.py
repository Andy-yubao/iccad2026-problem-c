"""One Node26 run -> freeze own-Loss best -> fixed500, no automatic retry."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import argparse
from datetime import datetime, timezone
from .io import OWNER, ROOT, save, exclusive, sources, sha

def now():return datetime.now(timezone.utc).isoformat()

def freeze_best():
    import torch
    run=OWNER/'runs/node26_seed0'
    result=json.loads((run/'results.json').read_text(encoding='utf-8'))
    if result['status'] not in ('early_stop','budget','max_epochs') or result.get('error'):
        raise RuntimeError('Training did not finish successfully')
    path=run/'checkpoints/best.pt'
    best=torch.load(path,map_location='cpu',weights_only=False)
    p=best['progress'];record=next(h for h in p['history'] if h['epoch']==p['best_epoch'])
    assert p['best']==result['best_fixed_loss'] and p['best_epoch']==result['best_epoch']
    assert best['identity']['sources']==sources()
    frozen=dict(checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),
                best_epoch=p['best_epoch'],own_selection_loss=p['best'],
                common_final_dev_loss=record['dev_monitor']['final_total'],
                common_final_train_loss=record['train_monitor']['final_total'],
                training_seconds=result['elapsed_seconds'],training_epochs=result['epochs'],
                training_steps=result['steps'],parameters=sum(t.numel() for t in best['model'].values()),
                frozen_before_Q=True,frozen_at=now())
    save(OWNER/'evaluation/frozen_model.json',frozen)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorized-recovery',action='store_true');args=parser.parse_args()
    with exclusive(OWNER/'queue.lock'):
        if (OWNER/'queue_state.json').exists():
            old=json.loads((OWNER/'queue_state.json').read_text(encoding='utf-8'))
            if not args.authorized_recovery or old.get('status')!='needs_attention' or old.get('recovery_id'):
                raise FileExistsError('Queue already dispatched; no automatic retry')
        auth=json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
        if not (auth.get('training_enabled') and auth.get('automatic_evaluation') and auth.get('automatic_queue')):
            raise PermissionError('Missing authorization')
        if auth['allowed_arms']!=['node26']:raise PermissionError('Unexpected candidate')
        if args.authorized_recovery and not auth.get('recovery_20261109_authorized'):
            raise PermissionError('Explicit recovery authorization missing')
        gate=json.loads((OWNER/'preflight.json').read_text(encoding='utf-8'))
        if gate['status']!='passed' or gate['sources']!=sources():raise RuntimeError('Implementation/source gate mismatch')
        state=dict(status='running',pid=os.getpid(),started_at=now(),active=None,completed=[],test_read=False,automatic_evaluation=True,
                   recovery_id='20261109_1' if args.authorized_recovery else None)
        requested=[False];signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        child=None
        def stopped():
            return requested[0] or any((OWNER/p).exists() for p in ('STOP','QUEUE_STOP','runs/node26_seed0/STOP','evaluation/STOP'))
        def beat(**kw):state.update(kw);save(OWNER/'queue_state.json',dict(**state,updated_at=now()))
        def launch(stage,module,stage_args):
            nonlocal child
            if stopped():raise InterruptedError('STOP; no next stage')
            logs=OWNER/'launch_logs'/('recovery1' if args.authorized_recovery else 'initial');logs.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-B','-m',f'experiments.E117_node26.{module}',*stage_args]
            with (logs/f'{stage}.stdout.log').open('x',encoding='utf-8') as out,(logs/f'{stage}.stderr.log').open('x',encoding='utf-8') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                beat(active=dict(stage=stage,pid=child.pid,started_at=now()))
                while child.poll() is None:
                    if stopped():(OWNER/'STOP').write_text('Save state; no next stage.\n',encoding='utf-8')
                    beat();time.sleep(5)
            if stopped():raise InterruptedError('Stopped; no next stage')
            if child.returncode!=0:raise RuntimeError(f'{stage} failed; no retry')
            state['completed'].append(dict(stage=stage,completed_at=now()));child=None;beat(active=None)
        try:
            launch('node26','train',['--arm','node26',*(['--resume','--authorized-recovery'] if args.authorized_recovery else [])])
            if stopped():raise InterruptedError('Stopped before freeze')
            freeze_best()
            launch('evaluation','evaluation.run',['--arm','node26'])
            result=json.loads((OWNER/'runs/node26_seed0/evaluation_no_mib_dev500/results.json').read_text(encoding='utf-8'))
            if result['status']!='complete' or result['count']!=500:raise RuntimeError('Incomplete evaluation')
            launch('train5_layers','diagnostics',[])
            beat(status='complete',active=None,completed_at=now())
        except BaseException as error:
            if child is not None and child.poll() is None:(OWNER/'STOP').write_text('Controller stopped; save and exit.\n',encoding='utf-8')
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention',error=repr(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
