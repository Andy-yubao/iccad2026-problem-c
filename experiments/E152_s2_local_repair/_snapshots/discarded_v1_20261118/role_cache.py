"""Read-only reuse of complete, source/member checked E149 public-role sidecars."""
import hashlib
import json
from pathlib import Path
import torch
from datasets.random_reference import RandomReference
from .io import HERE,ROOT,sha

def attach(rows,role,indices,plan,halted,beat):
    dataset=RandomReference(role)
    ids=[dataset.indices[i] for i in (range(len(dataset)) if indices is None else indices)]
    identity=dict(schema='E149.public_roles.v1',base_cache_key=plan['required_cache_key'],producer_sha256=sha(HERE/'role_features.py'),
        shortest_path='Exact unweighted positive B2B/P2B graph, group separate',ports='Unique referenced coordinates; coincident ports merged by weight sum',dtype='float32 features/edges, int16 hops; no labels')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    cache_root=(ROOT/plan['role_cache_root']).resolve()
    if not cache_root.is_relative_to(ROOT):raise ValueError('Role cache outside workspace')
    base=cache_root/key/dataset.role
    if not base.is_dir():raise RuntimeError('Registered existing role cache missing')
    meta=base.parent/'identity.json'
    if meta.exists():assert json.loads(meta.read_text(encoding='utf-8'))==identity
    else:raise RuntimeError('Existing role cache identity missing')
    groups={}
    for j,i in enumerate(ids):groups.setdefault(i//256,[]).append((j,i))
    output=[None]*len(rows);built=reused=0
    for block,entries in groups.items():
        if halted():return None
        path=base/f'{block:04d}.pt';manifest=path.with_suffix('.json');content={}
        if path.exists():
            m=json.loads(manifest.read_text(encoding='utf-8'));assert m['identity']==key and m['sha256']==sha(path)
            content=torch.load(path,map_location='cpu',weights_only=False)
        if not path.exists():raise RuntimeError('Existing role cache chunk missing')
        for j,index in entries:
            if index not in content:
                raise RuntimeError('Complete existing role cache required; no rebuilding')
            else:reused+=1
            assert content[index]['member']==list(rows[j]['member'])==dataset.records[index]['member']
            output[j]=dict(rows[j],role=content[index]['role'])
        beat(activity='preparing/reusing public role sidecars',role_cache_built=built,role_cache_reused=reused,role_cache_loaded=built+reused,role_cache_count=len(rows))
    return output
