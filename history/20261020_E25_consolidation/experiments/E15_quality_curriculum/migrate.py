"""Prepare an explicit v1->v2 continuation; never update the source checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile
import torch
from .io import HERE, ROOT, sha, save, snapshot, checkpoint, sources, exclusive
from .model import GeometryModel
from .train import make_identity, seed_all


def prepare_continuation(source, output, device='cuda'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or not output.is_relative_to(HERE/'runs'):
        raise ValueError('Continuation must have a distinct E15/runs directory')
    with exclusive(output.with_suffix('.lock')):
        original_sha = sha(source/'checkpoints/latest.pt')
        if (output/'continuation.json').exists():
            existing = json.loads((output/'continuation.json').read_text(encoding='utf-8'))
            if existing['source_checkpoint_sha256'] != original_sha:
                raise ValueError('Continuation already prepared from a different checkpoint')
            print('Continuation already prepared:', output, flush=True)
            return
        old = torch.load(source/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        if old['finished'] or old['status'] != 'user_stop' or old['identity']['arm'] != 'supervised_small':
            raise ValueError('Only the explicitly paused v1 supervised run is supported')
        if old['identity'].get('revision'):
            raise ValueError('Use normal --resume for a v2 checkpoint')
        with zipfile.ZipFile(source/'source_snapshot.zip') as archive:
            for name, digest in old['identity']['sources'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError('Original source snapshot checksum mismatch')
        # Shape representation and public feature math remain byte-identical.
        for name in ('model.py', 'features.py'):
            relative = 'experiments/E15_quality_curriculum/'+name
            if old['identity']['sources'][relative] != sha(HERE/name):
                raise ValueError('Model/feature math changed; transfer is not equivalent')
        cfg = old['identity']
        args = argparse.Namespace(arm=cfg['arm'], device=device, seed=cfg['seed'],
            max_hours=cfg['max_seconds']/3600, max_epochs=cfg['max_epochs'], batch_size=cfg['batch_size'])
        if device != cfg['device']:
            raise ValueError('Keep the existing device for this continuation')
        identity = make_identity(args)
        for key in ('registry_sha256','paths_sha256','width','layers','seed','batch_size','max_seconds','max_epochs','torch','numpy','cuda'):
            if identity[key] != cfg[key]: raise ValueError('Continuation identity mismatch: '+key)
        model = GeometryModel(**old['model_config'])
        model.load_state_dict(old['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        optimizer.load_state_dict(old['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3,
                            threshold=1e-5,threshold_mode='abs',min_lr=1e-6)
        new = copy.deepcopy(old)
        p = new['progress']
        p['legacy_history'] = p.pop('history')
        p.update(history=[], best_proxy=None, best_epoch=None, bad_checks=0, pending_monitor=True,
                 full_dev_complete=False)
        p.pop('best_rank',None)
        ledger = json.loads((source/'budget.json').read_text(encoding='utf-8'))
        new.update(identity=identity, elapsed_seconds=max(old['elapsed_seconds'],ledger['elapsed_seconds']),
                   scheduler=scheduler.state_dict(), status='continuation_prepared', finished=False)
        output.mkdir(parents=True,exist_ok=False)
        checkpoint(output/'checkpoints/latest.pt',new)
        snapshot(output/'source_snapshot.zip',identity)
        partial = source/f'quality/epoch_{p["epoch"]:04d}.partial.json'
        prior_exposure = len(json.loads(partial.read_text(encoding='utf-8'))) if partial.exists() else 0
        report = {'source_run':str(source),'source_checkpoint_sha256':original_sha,
            'source_identity':cfg,'new_identity':identity,'epoch':p['epoch'],'step':p['step'],
            'cursor':p['cursor'],'elapsed_seconds_carried':new['elapsed_seconds'],
            'model_optimizer_rng_order_seen_preserved':True,
            'change':'Cached public inputs; no train-time repair; fixed proxy-based selection and scheduler reset',
            'old_postprocessed_dev_examples':prior_exposure,
            'old_partial_quality_not_reused':True,'original_run_unchanged':True,'test_read':False}
        save(output/'continuation.json',report)
        save(output/'config.json',{'identity':identity,'continuation':report,
            'command':'Prepared only. User must run launcher -Resume to continue.'})
        save(output/'budget.json',{'elapsed_seconds':new['elapsed_seconds']})
        save(output/'progress.json',{'status':new['status'],'finished':False,'epoch':p['epoch'],
            'step':p['step'],'cursor':p['cursor'],'elapsed_seconds':new['elapsed_seconds'],'postprocessing_calls':0})
        assert sha(source/'checkpoints/latest.pt') == original_sha
        print(f'Prepared continuation at epoch {p["epoch"]}, step {p["step"]}; inherited {new["elapsed_seconds"]/3600:.3f} h',flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    a=ap.parse_args();prepare_continuation(a.source,a.output,a.device)
