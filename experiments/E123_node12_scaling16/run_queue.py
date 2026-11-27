"""Three authorized scratch36k depth16 candidates, serial train/dev500; stop on error."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from .io import ROOT,OWNER,save,sha,exclusive

ENTRIES=(('E122_global2_scaling16', 'global2'), ('E123_node12_scaling16', 'node12'), ('E124_residual_scaling16', 'residual'))

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def verify(owner,arm):
    auth=read(owner/'authorization.json');gate=read(owner/'preflight.json')
    if not (auth.get('training_enabled') and auth.get('automatic_evaluation') and auth.get('automatic_queue')):
        raise PermissionError('Candidate does not have current training/evaluation authorization')
    if auth['allowed_arms']!=[arm] or gate['status']!='passed':raise ValueError('Candidate/gate scope mismatch')
    for name,digest in gate['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Candidate sources changed after implementation check: '+name)

def freeze(owner,arm):
    import torch
    run=owner/'runs'/f'{arm}_seed0';result=read(run/'results.json')
    if result.get('status') not in ('early_stop','max_epochs','budget') or result.get('error'):
        raise RuntimeError('Training did not finish successfully; no evaluation or next candidate')
    path=run/'checkpoints/best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    progress=state['progress'];entry=next(h for h in progress['history'] if h['epoch']==progress['best_epoch'])
    if progress['best']!=result['best_fixed_loss']:raise ValueError('Best identity mismatch')
    for name,digest in state['identity']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Recorded training source changed')
    save(owner/'evaluation/frozen_model.json',dict(checkpoint=path.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(path),best_epoch=progress['best_epoch'],own_selection_loss=progress['best'],
        common_final_dev_loss=entry['dev_monitor']['final_total'],common_final_train_loss=entry['train_monitor']['final_total'],
        training_seconds=result['elapsed_seconds'],training_epochs=result['epochs'],training_steps=result['steps'],
        parameters=sum(v.numel() for v in state['model'].values()),frozen_before_Q=True,frozen_at=now()))

def main():
    with exclusive(OWNER/'queue.lock'):
        if (OWNER/'queue_state.json').exists():raise FileExistsError('Already dispatched; no automatic retry')
        for name,arm in ENTRIES:verify(ROOT/'experiments'/name,arm)
        state=dict(status='running',pid=os.getpid(),started_at=now(),completed=[],active=None,
                   order=[name for name,_ in ENTRIES],test_read=False,automatic_evaluation=True,automatic_retry=False)
        requested=[False];signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        child=None;active_owner=None
        def stopped(owner):
            arm=next(a for name,a in ENTRIES if name==owner.name)
            return requested[0] or any((p/f).exists() for p in (OWNER,owner) for f in ('STOP','QUEUE_STOP')) or any(
                (owner/f).exists() for f in (f'runs/{arm}_seed0/STOP','evaluation/STOP'))
        def beat(**fields):state.update(fields);save(OWNER/'queue_state.json',dict(**state,updated_at=now()))
        def launch(owner,arm,phase):
            nonlocal child
            if stopped(owner):raise InterruptedError('STOP before next stage')
            logs=owner/'launch_logs';logs.mkdir(exist_ok=True)
            module='train' if phase=='train' else 'evaluation.run'
            command=[sys.executable,'-u','-B','-m',f'experiments.{owner.name}.{module}','--arm',arm]
            with (logs/f'{phase}.stdout.log').open('x',encoding='utf-8') as out,(logs/f'{phase}.stderr.log').open('x',encoding='utf-8') as err:
                child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,
                    creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0)
                beat(active=dict(experiment=owner.name,phase=phase,pid=child.pid,started_at=now()))
                if owner!=OWNER:save(owner/'queue_state.json',dict(status='running',controller=str(OWNER.relative_to(ROOT)),active=state['active']))
                while child.poll() is None:
                    if stopped(owner):(owner/'STOP').write_text('Controller STOP; save full state; no next stage.\n',encoding='utf-8')
                    beat();time.sleep(5)
            if stopped(owner):raise InterruptedError('Stopped; no next stage')
            if child.returncode!=0:raise RuntimeError(f'{owner.name}/{phase} failed; no retry or next stage')
            state['completed'].append(dict(experiment=owner.name,phase=phase,completed_at=now()))
            child=None;beat(active=None)
        try:
            for name,arm in ENTRIES:
                active_owner=ROOT/'experiments'/name
                if stopped(active_owner):raise InterruptedError('STOP before candidate')
                verify(active_owner,arm)
                launch(active_owner,arm,'train')
                if stopped(active_owner):raise InterruptedError('STOP before best freeze')
                freeze(active_owner,arm)
                launch(active_owner,arm,'evaluation')
                result=read(active_owner/'runs'/f'{arm}_seed0/evaluation_no_mib_dev500/results.json')
                if result.get('status')!='complete' or result.get('count')!=500:raise RuntimeError('Incomplete dev500 evaluation')
                if active_owner!=OWNER:save(active_owner/'queue_state.json',dict(status='complete',completed_at=now(),test_read=False))
            beat(status='complete',active=None,completed_at=now())
        except BaseException as error:
            if child is not None and child.poll() is None and active_owner:
                (active_owner/'STOP').write_text('Controller exit; save full state; no continuation.\n',encoding='utf-8')
            status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention'
            beat(status=status,error=repr(error),traceback=traceback.format_exc())
            if active_owner and active_owner!=OWNER:save(active_owner/'queue_state.json',dict(status=status,error=repr(error),test_read=False))
            raise

if __name__=='__main__':main()
