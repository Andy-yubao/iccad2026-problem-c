"""Four-process resumable tensor preparation for E10 reference/contact losses."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import torch
from datasets.reference_layouts import ReferenceLayouts, REGISTRY
from .io import ROOT, HERE, save, sha, checkpoint, exclusive

CACHE = HERE/'runs/prepared_v5_e10_cache_v1'


def identity():
    return {'format': 'E17.original_reference_tensors.v1', 'registry': sha(REGISTRY),
        'paths': sha(ROOT/'protocol/paths.json'),
        'sources': {name: sha(HERE/name) for name in ('cache.py','data.py','features.py','batches.py')},
        'datasets_sources': {p.name: sha(p) for p in sorted((ROOT/'datasets').glob('*.py'))}}


def verify(role):
    if role not in ('train','dev'): raise PermissionError('Test cache is forbidden')
    folder = CACHE/role
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity(): raise ValueError('Cache source/data identity changed')
    expected = [list(r['member']) for r in ReferenceLayouts(role).records]
    if manifest['members'] != expected: raise ValueError('Prepared membership changed')
    for item in manifest['parts']:
        stat = (folder/item['name']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('Prepared chunk changed')
    return manifest


def initialize(role):
    global dataset
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    dataset = ReferenceLayouts(role)


def prepare_part(start, ids, path):
    from .data import prepare
    value = {'identity': identity(), 'indices': ids, 'rows': [prepare(dataset[i]) for i in ids]}
    checkpoint(path, value)
    return {'start':start,'count':len(ids),'pid':os.getpid(),'torch_threads':torch.get_num_threads()}


def build(role, stopped):
    data = ReferenceLayouts(role); folder = CACHE/role; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'manifest.json').exists():
        verify(role); print(f'Prepared cache ready: {role}', flush=True); return True
    stamp = folder/'identity.json'; current = identity()
    if stamp.exists() and json.loads(stamp.read_text()) != current:
        raise ValueError('Interrupted cache identity differs')
    save(stamp,current)
    ids = sorted(range(len(data)), key=lambda i:data.records[i]['member'])
    tasks = [(start,ids[start:start+128],str(folder/f'{start:06d}.pt')) for start in range(0,len(ids),128)]
    todo = [t for t in tasks if not Path(t[2]).exists()]
    # Existing chunks are committed atomically by this exact preparation identity.
    complete = len(data)-sum(len(t[1]) for t in todo)
    print(f'Prepare {role}: reuse {complete}/{len(data)}; four processes, one thread each',flush=True)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(role,)) as pool:
        pending = {}; remaining=iter(todo)
        def submit():
            if stopped(): return
            task=next(remaining,None)
            if task is not None: pending[pool.submit(prepare_part,*task)]=task
        for _ in range(4): submit()
        while pending:
            done,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future); result=future.result()
                complete+=len(task[1]); save(folder/'progress.json',{'completed':complete,'total':len(data),'last_worker':result})
                print(f'Prepare {role}: {complete}/{len(data)}',flush=True); submit()
    if complete != len(data) or stopped(): return False
    parts=[]
    for _,_,name in tasks:
        path=Path(name); stat=path.stat()
        parts.append({'name':path.name,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    save(folder/'manifest.json',{'identity':current,'count':len(data),'members':[r['member'] for r in data.records],
        'parts':parts,'workers':4,'threads_per_worker':1,'test_read':False})
    return True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
    CACHE.mkdir(parents=True,exist_ok=True)
    requested=[False]; signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    stopped=lambda:requested[0] or (CACHE/'STOP').exists() or (HERE/'runs/20261020_e10_geometry_v5_seed0/STOP').exists()
    with exclusive(CACHE.with_suffix('.lock')):
        for role in ('train','dev'):
            if stopped() or not build(role,stopped): return
        save(CACHE/'results.json',{'status':'complete','test_read':False,'identity':identity()})


if __name__=='__main__': main()
