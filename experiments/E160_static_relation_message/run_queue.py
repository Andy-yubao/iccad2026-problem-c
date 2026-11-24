"""Prepared single-candidate queue; formal training requires explicit activation."""
import json,subprocess,sys,os
from .io import HERE,ROOT,save,sha,sources,exclusive
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def freeze(arm):
    import torch
    run=HERE/'runs'/f'{arm}_seed0';result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result['error']:raise RuntimeError('Stage did not finish successfully')
    path=run/'checkpoints/best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['identity']['sources']==sources()
    save(HERE/'frozen_initial.json',dict(arm=arm,checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),best_epoch=state['progress']['best_epoch'],selection_loss=state['progress']['best']))
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true',required=True);p.parse_args()
    auth=read(HERE/'authorization.json')
    if not auth['training_enabled']:raise PermissionError('Preparation is authorized; formal training has not been activated')
    gate=read(HERE/'preparation.json')
    if gate['status']!='prepared' or gate['sources']!=sources():raise RuntimeError('Preparation gate/source mismatch')
    with exclusive(HERE/'queue.lock'):
        state=read(HERE/'queue_state.json')
        if state['status']!='prepared':raise RuntimeError('Not a fresh queue')
        state.update(status='running',pid=os.getpid());save(HERE/'queue_state.json',state)
        for arm in ('e160_initial',):
            if (HERE/'STOP').exists() or (HERE/'QUEUE_STOP').exists():break
            state.update(active=arm);save(HERE/'queue_state.json',state)
            command=[sys.executable,'-u','-B','-m','experiments.E160_static_relation_message.train','--arm',arm]
            code=subprocess.call(command,cwd=ROOT)
            if code or read(HERE/'runs'/f'{arm}_seed0/results.json')['status']=='user_stop':
                state.update(status='stopped',active=None);save(HERE/'queue_state.json',state);return
            freeze(arm);state.setdefault('completed',[]).append(arm);save(HERE/'queue_state.json',state)
        else:
            from .compare_loss import compare
            compare()
            state.update(status='complete',active=None,backend_training_started=False,postprocessing_calls=0);save(HERE/'queue_state.json',state);return
        state.update(status='user_stop',active=None);save(HERE/'queue_state.json',state)
if __name__=='__main__':main()
