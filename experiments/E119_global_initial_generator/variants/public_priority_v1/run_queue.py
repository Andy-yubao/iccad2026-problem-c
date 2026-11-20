"""One authorized train -> freeze own-new-Loss best -> original dev500."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from .io import ROOT,OWNER,save,sha,exclusive,sources
MODULE='experiments.E119_global_initial_generator.variants.public_priority_v1'
ARM='global2'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def verify():
    auth=read(OWNER/'authorization.json');gate=read(OWNER/'preflight.json')
    assert auth['training_enabled'] and auth['automatic_evaluation'] and auth['automatic_queue']
    assert auth['scope']=='E119.public_priority.v1.seed0' and auth['allowed_arms']==[ARM]
    assert gate['status']=='passed' and gate['sources']==sources()
    assert sha(OWNER/'preflight.json')==auth['preflight_sha256']
    assert not auth['retry'] and not auth['test_read']

def freeze():
    import torch
    run=OWNER/'runs/global2_seed0';result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result.get('error'):raise RuntimeError('Unsuccessful training; evaluation blocked')
    path=run/'checkpoints/best.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    p=checkpoint['progress'];entry=next(x for x in p['history'] if x['epoch']==p['best_epoch'])
    assert p['best']==result['best_fixed_loss']
    for name,digest in checkpoint['identity']['sources'].items():assert sha(ROOT/name)==digest
    save(OWNER/'evaluation/frozen_model.json',dict(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),
        best_epoch=p['best_epoch'],own_selection_loss=p['best'],
        common_final_dev_loss=entry['dev_monitor']['original_final_total'],
        common_final_train_loss=entry['train_monitor']['original_final_total'],
        own_new_final_dev_loss=entry['dev_monitor']['final_total'],
        training_seconds=result['elapsed_seconds'],training_epochs=result['epochs'],training_steps=result['steps'],
        parameters=sum(v.numel() for v in checkpoint['model'].values()),frozen_before_Q=True,frozen_at=now()))

def main():
    with exclusive(OWNER/'queue.lock'):
        if (OWNER/'queue_state.json').exists():raise FileExistsError('Already dispatched; no retry')
        verify();requested=[False];signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        child=None
        state=dict(status='running',pid=os.getpid(),started_at=now(),active=None,completed=[],
            order=['train','freeze_unique_own_new_loss_best','original_dev500'],automatic_evaluation=True,automatic_retry=False,test_read=False)
        def beat(**kw):state.update(kw);save(OWNER/'queue_state.json',dict(**state,updated_at=now()))
        def stopped():return requested[0] or any((OWNER/p).exists() for p in ('STOP','QUEUE_STOP','runs/global2_seed0/STOP','evaluation/STOP','runs/global2_seed0/evaluation_no_mib_dev500/STOP'))
        def launch(phase):
            nonlocal child
            if stopped():raise InterruptedError('STOP before stage')
            verify();module=MODULE+('.train' if phase=='train' else '.evaluation.run')
            logs=OWNER/'launch_logs';logs.mkdir(exist_ok=True)
            with (logs/(phase+'.stdout.log')).open('x',encoding='utf-8') as out,(logs/(phase+'.stderr.log')).open('x',encoding='utf-8') as err:
                flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
                child=subprocess.Popen([sys.executable,'-u','-B','-m',module,'--arm',ARM],cwd=ROOT,stdout=out,stderr=err,creationflags=flags)
                beat(active=dict(phase=phase,pid=child.pid,started_at=now()))
                while child.poll() is None:
                    if stopped():(OWNER/'STOP').write_text('User stop; save complete accepted state and do not advance.\n',encoding='utf-8')
                    beat();time.sleep(5)
            if stopped():raise InterruptedError('Stopped; next stage blocked')
            if child.returncode!=0:raise RuntimeError(f'{phase} failed with code {child.returncode}; no retry')
            state['completed'].append(dict(phase=phase,completed_at=now()));child=None;beat(active=None)
        try:
            beat();launch('train')
            if stopped():raise InterruptedError('STOP before best freeze')
            freeze();state['completed'].append(dict(phase='freeze_unique_own_new_loss_best',completed_at=now()));beat()
            launch('evaluation')
            result=read(OWNER/'runs/global2_seed0/evaluation_no_mib_dev500/results.json')
            assert result['status']=='complete' and result['count']==500
            beat(status='complete',active=None,completed_at=now())
        except BaseException as error:
            if child is not None and child.poll() is None:(OWNER/'STOP').write_text('Controller exited; save full state; no continuation.\n',encoding='utf-8')
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention',error=repr(error),traceback=traceback.format_exc())
            raise
if __name__=='__main__':main()
