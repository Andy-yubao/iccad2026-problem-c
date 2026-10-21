"""One resumable frozen-best dev500 postprocessing pass, after training only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import torch
from .io import HERE, sha, save, sources, snapshot, exclusive
from .train import seed_all, REVISION
from .model import GeometryModel
from .data import load_rows
from .cache import identity as cache_identity
from .evaluation import quality_monitor, summary


def freeze_once(folder, frozen, resume=False):
    """False means a completed pass; never permit another candidate."""
    if folder.exists():
        if json.loads((folder/'frozen.json').read_text(encoding='utf-8')) != frozen:
            raise ValueError('This run already froze a different evaluation; no second candidate allowed')
        if (folder/'results.json').exists():
            return False
        if not resume:
            raise ValueError('Use --resume for the same interrupted frozen pass')
    else:
        folder.mkdir(parents=True)
        save(folder/'frozen.json', frozen)
    return True


def validate_dependencies(identity, current):
    # LR/control changes do not change the old model's inference contract.
    prefix = 'experiments/E27_overlap_weight/'
    required = {prefix+name for name in ('model.py','features.py','batches.py','data.py','cache.py','evaluation.py','io.py')}
    required.update(name for name in current if name.startswith(
        (prefix+'backend/', 'datasets/', 'scoring/', 'third_party/')))
    if any(identity['sources'].get(name) != current[name] for name in required):
        raise ValueError('Inference/data/scorer/postprocessor changed since training')
    if identity['cache_identity'] != cache_identity():
        raise ValueError('Frozen data identity changed since training')


def initialize_worker(path, seed):
    global worker_model, worker_rows
    # Ctrl+C is handled by the parent; finish bounded in-flight cases durably.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    seed_all(seed, threads=2)
    torch.set_num_interop_threads(1)
    state = torch.load(path, map_location='cpu', weights_only=False)
    worker_model = GeometryModel(**state['model_config'])
    worker_model.load_state_dict(state['model'])
    worker_rows = load_rows('dev_monitor500')


def evaluate_index(index):
    # Case-local seeds also make out-of-order completion/resume reproducible.
    seed_all(worker_seed+index, threads=2)
    return index, quality_monitor(worker_model, [worker_rows[index]], 'cpu')['records'][0]


def initialize_seeded_worker(path, seed):
    global worker_seed
    worker_seed = seed
    initialize_worker(path, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--workers', type=int, default=2, help='CPU processes, each with two library threads')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--finalize-stopped', action='store_true',help='User-authorized best-only assessment after stopped training; preserve original checkpoints')
    args = ap.parse_args(); args.run = args.run.resolve()
    if not args.run.is_relative_to(HERE/'runs'): raise ValueError('Not an E27 run')
    if not 1 <= args.workers <= 8: raise ValueError('Use 1..8 workers')
    cpu = args.device == 'cpu'
    workers = args.workers if cpu else 1
    # Spawn inherits limits before importing numpy/torch in the child process.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    seed_all(0, threads=2)
    folder = args.run/'final_evaluation'
    with exclusive(args.run.with_suffix('.lock')), exclusive(folder.with_suffix('.lock')):
        latest = torch.load(args.run/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        identity = latest['identity']
        finalization=None
        if not latest['finished'] and args.finalize_stopped:
            if latest.get('status')!='user_stop':raise ValueError('Only explicitly user-stopped training may be finalized')
            state=latest['scheduler'];epoch=latest['progress']['epoch']
            if not (state['early_bad']>=3 and state['drops']==3 and epoch>=state['last_drop_epoch']+2):
                raise ValueError('New stop3/minimum2 rule has not been satisfied')
            finalization={'reason':'User requested direct verification after changing early-stop rule',
                'original_training_status':latest['status'],'original_finished_flag':latest['finished'],
                'completed_epochs':epoch,'best_epoch':latest['progress']['best_epoch'],
                'early_bad':state['early_bad'],'lr_drops':state['drops'],'last_drop_epoch':state['last_drop_epoch'],
                'new_rule':{'early_bad_min':3,'lr_drops_required':3,'lowest_completed_epochs_min':2},
                'training_checkpoint_modified':False,'full_dev_complete':latest['progress']['full_dev_complete']}
        if (not latest['finished'] and finalization is None) or identity.get('revision') != REVISION:
            raise ValueError('Finish or explicitly finalize training before final postprocessing')
        if identity['implementation_check']: raise ValueError('Not a formal model')
        path = args.run/'checkpoints/best.pt'; digest = sha(path)
        best = torch.load(path, map_location='cpu', weights_only=False)
        if best['identity'] != identity: raise ValueError('Best/latest identities differ')
        if finalization is not None:
            finalization['checkpoint_sha256']=digest
            marker=args.run/'verification_finalization.json'
            if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=finalization:raise ValueError('Stopped-run finalization identity changed')
            save(marker,finalization)
        current = sources()
        validate_dependencies(identity, current)
        rows = load_rows('dev_monitor500')
        if len(rows) != 500: raise ValueError('The frozen final panel must have 500 cases')
        members = [list(row['member']) for row in rows]
        frozen = {'checkpoint_sha256':digest, 'identity':identity, 'panel':'dev_monitor500',
            'members':members, 'postprocess':{'normal_seconds':5., 'extra_rescue_seconds':16.},
            'execution':{'device':args.device, 'workers':workers, 'threads_per_worker':2,
                         'case_seed':'training_seed + panel_index'}, 'evaluation_sources':current}
        if finalization is not None:frozen['training_finalization']=finalization
        if not freeze_once(folder, frozen, args.resume):
            print('Final pass already completed; reusing stored result:',folder,flush=True); return
        if not (folder/'config.json').exists():
            save(folder/'config.json', {'command':[sys.executable,*sys.argv], 'frozen':frozen,
                'timing_scope':'Concurrent CPU pass; timings under shared load, not isolated speed benchmarks',
                'source_scope':'Training identity preserved; snapshot contains current evaluation sources'})
            snapshot(folder/'source_snapshot.zip', {'sources':current})
        partial = folder/'partial.json'
        entries = json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else []
        completed = {}
        for entry in entries:
            index, record = entry['index'], entry['record']
            if index in completed or not 0 <= index < 500 or record['member'] != members[index]:
                raise ValueError('Partial evaluation panel mismatch')
            completed[index] = record
        requested = [False]
        handler = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0,True))
        begin = time.perf_counter()
        def stopped(): return requested[0] or (folder/'STOP').exists()
        def persist():
            save(partial, [{'index':i,'record':completed[i]} for i in sorted(completed)])
            save(folder/'progress.json', {'completed':len(completed),'total':500,
                'checkpoint_sha256':digest,'elapsed_seconds_this_invocation':time.perf_counter()-begin,
                'stopped':stopped()})
            print(f'Final quality evaluation: {len(completed)}/500',flush=True)
        try:
            pending_ids = iter(i for i in range(500) if i not in completed)
            if cpu:
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                        initializer=initialize_seeded_worker, initargs=(path,identity['seed'])) as pool:
                    pending = {}
                    def submit_one():
                        if stopped(): return
                        index = next(pending_ids, None)
                        if index is not None: pending[pool.submit(evaluate_index,index)] = index
                    for _ in range(workers): submit_one()
                    while pending:
                        done, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            index, record = future.result()
                            if pending.pop(future) != index: raise ValueError('Worker index mismatch')
                            completed[index] = record
                            persist(); submit_one()
            else:
                model = GeometryModel(**best['model_config']).to(args.device)
                model.load_state_dict(best['model'])
                for index in pending_ids:
                    if stopped(): break
                    seed_all(identity['seed']+index, threads=2)
                    completed[index] = quality_monitor(model,[rows[index]],args.device)['records'][0]
                    persist()
            if len(completed) != 500:
                persist()
                print('Paused; --resume skips completed cases',flush=True); return
            records = [completed[i] for i in range(500)]
            metrics = summary(records)
            metrics['raw_hard_legal'] = sum(r.get('raw_metrics',{}).get('is_feasible',False) for r in records)
            result = {'summary':metrics,'records':records,'status':'complete','checkpoint_sha256':digest,
                'scope':'One frozen best, dev500 after training; not used for selection',
                'test_read':False,'tail_test_read':False}
            save(folder/'results.json',result)
            save(HERE/'runs/20261020_pipeline/pipeline.json',{'stage':'final_dev500','status':'complete',
                'checkpoint_sha256':digest,'test_read':False})
            if partial.exists(): partial.unlink()
            print(json.dumps(metrics,ensure_ascii=False),flush=True)
        finally: signal.signal(signal.SIGINT,handler)


if __name__ == '__main__': main()
