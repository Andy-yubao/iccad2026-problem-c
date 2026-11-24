"""Explicitly activated sequential scratch stages; preparation never starts work."""
import argparse,json,os,subprocess,sys
from .io import HERE,ROOT,save,sha,sources,exclusive
from .loss_config import ARMS


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def freeze(arm):
    import torch
    run=HERE/'runs'/f'{arm}_seed0';result=read(run/'results.json')
    if result['status'] not in ('early_stop','budget','max_epochs') or result['error']:
        raise RuntimeError('Unsuccessful stage blocks the queue')
    path=run/'checkpoints/best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']['sources']!=sources():raise ValueError('Stage sources changed')
    frozen=dict(arm=arm,checkpoint=path.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(path),
        best_epoch=state['progress']['best_epoch'],own_selection_loss=state['progress']['best'],
        parameters=read(run/'parameter_counts.json')['total'],training_seconds=result['elapsed_seconds'],
        training_epochs=result['epochs'],common_final_dev_loss=state['progress']['best'],
        own_new_final_dev_loss=state['progress']['best'],selection_frozen_before_Q=True)
    save(HERE/f'frozen_{arm}.json',frozen)
    return frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--start',action='store_true',required=True);parser.parse_args()
    auth=read(HERE/'authorization.json')
    if not auth['training_enabled']:raise PermissionError('E161 is prepared only; training is disabled')
    prep=read(HERE/'preparation.json')
    if prep['status']!='prepared' or prep['sources']!=sources():raise ValueError('Prepared source identity changed')
    with exclusive(HERE/'queue.lock'):
        state=read(HERE/'queue_state.json')
        if state['status']!='prepared':raise RuntimeError('Only the registered fresh queue is allowed')
        state.update(status='running',pid=os.getpid());save(HERE/'queue_state.json',state)
        for arm in ARMS:
            if any((HERE/name).exists() for name in ('STOP','QUEUE_STOP','PAUSE_REQUEST')):
                state.update(status='user_stop',active=None);save(HERE/'queue_state.json',state);return
            state.update(active=arm);save(HERE/'queue_state.json',state)
            command=[sys.executable,'-u','-B','-m','experiments.E161_final_no_mib.train','--arm',arm]
            code=subprocess.call(command,cwd=ROOT)
            if code or read(HERE/'runs'/f'{arm}_seed0/results.json')['status']=='user_stop':
                state.update(status='stopped',active=None);save(HERE/'queue_state.json',state);return
            frozen=freeze(arm);state['completed'].append(arm);save(HERE/'queue_state.json',state)
        save(HERE/'evaluation/frozen.json',frozen)
        if auth.get('postprocessing_enabled'):
            state.update(active='dev2000');save(HERE/'queue_state.json',state)
            code=subprocess.call([sys.executable,'-u','-B','-m','experiments.E161_final_no_mib.evaluation.run','--arm','e161_s2'],cwd=ROOT)
            if code:
                state.update(status='evaluation_error',active=None);save(HERE/'queue_state.json',state);return
        state.update(status='complete',active=None);save(HERE/'queue_state.json',state)


if __name__=='__main__':main()
