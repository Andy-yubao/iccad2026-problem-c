"""Versioned CPU preprocessing cache shared by compatible train/dev readers.

Only requested members are built. No model, random augmentation, CUDA state,
or test payload belongs here. Chunks may hold a partial authorized dev panel.
"""
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager
import numpy as np
import torch
from .paths import ROOT

CHUNK_SIZE = 256

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for blob in iter(lambda:f.read(4*2**20),b''):h.update(blob)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)

@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def _init_worker():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)

def _build(task):
    index,record,folder,checksum,producer=task
    path=Path(folder)/record['payload']
    if sha(path)!=checksum:raise ValueError('Raw payload checksum changed: '+str(path))
    with np.load(path,allow_pickle=False) as item:
        inputs=(len(item['areas']),*[item[k].copy() for k in ('areas','b2b','p2b','pins','constraints','specs')])
        reference=torch.as_tensor(item['reference_xywh'].copy()).float()
    module=importlib.import_module(producer)
    row=module.make_row(inputs,reference,record['member'])
    # Plain NumPy IPC avoids PyTorch's Windows named shared-storage mappings,
    # which otherwise accumulate one persistent OS handle per small tensor.
    row['graph']={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in row['graph'].items()}
    return index,row

def load_rows(dataset,producer,identity,checksums,indices=None,halted=lambda:False,
              beat=lambda **kw:None,workers=4,cache_root=None):
    """Reuse checked CPU tensors, fill only requested missing members once."""
    if dataset.role not in ('train','dev'):raise PermissionError('Test cache forbidden')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    base=Path(cache_root) if cache_root else ROOT/'datasets/static_tensor_cache_v1'
    if not base.resolve().is_relative_to(ROOT):raise ValueError('Cache path escapes workspace')
    folder=base/key;folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/'identity.json'
    with lock(folder/'identity.lock'):
        if metadata.exists():
            if json.loads(metadata.read_text(encoding='utf-8'))!=identity:raise ValueError('Cache identity mismatch')
        else:atomic_json(metadata,identity)
    selected=list(range(len(dataset))) if indices is None else list(indices)
    parent_ids=[dataset.indices[i] for i in selected]
    groups={}
    for i in parent_ids:groups.setdefault(i//CHUNK_SIZE,[]).append(i)
    rows={};built=0;reused=0;pool=None;begin=time.perf_counter()
    try:
        for block,requested in groups.items():
            if halted():return None
            path=folder/dataset.role/f'{block:04d}.pt';manifest=path.with_suffix('.json')
            with lock(path.with_suffix('.lock')):
                entry={};content={}
                if path.exists() or manifest.exists():
                    if not (path.exists() and manifest.exists()):raise ValueError('Incomplete cache commit: '+str(path))
                    entry=json.loads(manifest.read_text(encoding='utf-8'))
                    if entry['identity']!=key or sha(path)!=entry['sha256']:raise ValueError('Invalid tensor cache: '+str(path))
                    content=torch.load(path,map_location='cpu',weights_only=False)
                missing=[i for i in dict.fromkeys(requested) if i not in content]
                if missing:
                    beat(activity='building persistent static tensor cache',cache_built=built,cache_reused=reused,cache_requested=len(parent_ids),cache_role=dataset.role)
                    tasks=[(i,dataset.records[i],str(dataset.folder),checksums[dataset.records[i]['id']],producer) for i in missing]
                    if workers>1 and pool is None:pool=multiprocessing.get_context('spawn').Pool(workers,initializer=_init_worker)
                    iterator=pool.imap(_build,tasks,chunksize=8) if pool else map(_build,tasks)
                    for index,row in iterator:
                        if halted():return None
                        row['graph']={k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in row['graph'].items()}
                        content[index]=row;built+=1
                    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');torch.save(content,tmp);os.replace(tmp,path)
                    atomic_json(manifest,dict(identity=key,sha256=sha(path),members={str(i):list(r['member']) for i,r in content.items()}))
                reused+=len(requested)-len(missing)
                for i in requested:
                    row=content[i]
                    if list(row['member'])!=list(dataset.records[i]['member']):raise ValueError('Cached member/order mismatch')
                    rows[i]=row
            beat(activity='loading persistent static tensor cache',cache_built=built,cache_reused=reused,
                 loaded=len(rows),count=len(parent_ids),cache_role=dataset.role,cache_key=key)
        beat(cache_seconds=time.perf_counter()-begin,cache_built=built,cache_reused=reused)
        return [rows[i] for i in parent_ids]
    finally:
        if pool:
            pool.terminate();pool.join()
