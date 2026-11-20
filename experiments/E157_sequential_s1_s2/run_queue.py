"""Sequential S1 -> freeze best -> S2 -> freeze best -> full dev500."""
import json,os,subprocess,sys
from .io import HERE,ROOT,save,sha,sources,exclusive

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def freeze(arm):
    import torch
    run=HERE/'runs'/f'{arm}_seed0';result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result['error']:raise RuntimeError('Unsuccessful stage: '+arm)
    path=run/'checkpoints/best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['identity']['sources']==sources() and state['identity']['arm']==arm
    descriptor=dict(arm=arm,checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),best_epoch=state['progress']['best_epoch'],selection_loss=state['progress']['best'])
    save(HERE/('frozen_s1.json' if arm=='e157_s1' else 'frozen_s2.json'),descriptor)
    return descriptor

def halted():return any((HERE/n).exists() for n in ('STOP','QUEUE_STOP','PAUSE_REQUEST'))
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true',required=True);p.parse_args()
    auth=read(HERE/'authorization.json');gate=read(HERE/'preflight.json')
    assert auth['training_enabled'] and auth['allowed_arms']==['e157_s1','e157_s2']
    assert gate['status']=='passed' and gate['sources']==sources()
    with exclusive(HERE/'queue.lock'):
        state=read(HERE/'queue_state.json');assert state['status']=='prepared'
        state.update(status='running',pid=os.getpid());save(HERE/'queue_state.json',state)
        for arm in ('e157_s1','e157_s2'):
            if halted():
                state.update(status='user_stop',active=None);save(HERE/'queue_state.json',state);return
            state.update(active=arm);save(HERE/'queue_state.json',state)
            code=subprocess.call([sys.executable,'-u','-B','-m','experiments.E157_sequential_s1_s2.train','--arm',arm],cwd=ROOT)
            path=HERE/'runs'/f'{arm}_seed0/results.json'
            result=read(path) if path.exists() else {}
            if code or result.get('error') or result.get('status')=='user_stop':
                state.update(status='user_stop' if result.get('status')=='user_stop' else 'needs_attention',active=None);save(HERE/'queue_state.json',state);return
            desc=freeze(arm);state['completed'].append(arm);save(HERE/'queue_state.json',state)
            print('[FROZEN] '+json.dumps(desc),flush=True)
        if halted():
            state.update(status='user_stop',active=None);save(HERE/'queue_state.json',state);return
        state.update(active='evaluation_dev500');save(HERE/'queue_state.json',state)
        code=subprocess.call([sys.executable,'-u','-B','-m','experiments.E157_sequential_s1_s2.evaluation.run'],cwd=ROOT)
        path=HERE/'runs/e157_s2_seed0/evaluation_mib_dev500/results.json'
        state.update(status='complete' if not code and path.exists() else 'needs_attention',active=None)
        save(HERE/'queue_state.json',state)
if __name__=='__main__':main()
