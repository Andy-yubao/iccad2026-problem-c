"""Record user-finalized training without changing best weights or training identity."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from .io import HERE, sha, save, checkpoint, exclusive


def close(run):
    run=Path(run).resolve()
    if not run.is_relative_to(HERE/'runs'):raise ValueError('Not an E15 run')
    with exclusive(run.with_suffix('.lock')):
        if (run/'training_closed.json').exists():return
        latest_path=run/'checkpoints/latest.pt';best_path=run/'checkpoints/best.pt'
        latest=torch.load(latest_path,map_location='cpu',weights_only=False)
        best=torch.load(best_path,map_location='cpu',weights_only=False)
        if latest['status']!='user_stop' or latest['finished']:
            raise ValueError('Stop and save the training before finalizing it')
        if latest['identity']!=best['identity']:raise ValueError('Checkpoint identities differ')
        if latest['identity']['arm']!='supervised_small':raise ValueError('Only the first model is being finalized')
        best_digest=sha(best_path)
        backup=run/'checkpoints/latest_before_user_finalized.pt'
        if backup.exists():raise FileExistsError(backup)
        shutil.copy2(latest_path,backup)
        for name in ('progress.json','results.json'):
            if (run/name).exists():shutil.copy2(run/name,run/(Path(name).stem+'_before_user_finalized.json'))
        p=latest['progress']
        report={'reason':'User explicitly ended the first model training',
            'completed_epochs':p['epoch'],'partial_epoch_cursor':p['cursor'],'step':p['step'],
            'elapsed_seconds':latest['elapsed_seconds'],'best_epoch':best['progress']['epoch'],
            'best_step':best['progress']['step'],'best_checkpoint_sha256':best_digest,
            'best_weights_unchanged':True,'training_identity_unchanged':True,'status':'user_finalized',
            'scope':'Use frozen existing best for the one dev500 final pass; never resume this model'}
        latest.update(status='user_finalized',finished=True)
        checkpoint(latest_path,latest)
        progress=json.loads((run/'progress.json').read_text(encoding='utf-8'))
        progress.update(status='user_finalized',finished=True)
        save(run/'progress.json',progress)
        results=json.loads((run/'results.json').read_text(encoding='utf-8')) if (run/'results.json').exists() else {}
        results.update(status='user_finalized',finished=True,user_finalization=report)
        save(run/'results.json',results);save(run/'training_closed.json',report)
        if sha(best_path)!=best_digest:raise AssertionError('Best weights changed')
        print(f'Closed first model at {p["epoch"]} completed epochs, step {p["step"]}; frozen best epoch {best["progress"]["epoch"]}',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    close(ap.parse_args().run)
