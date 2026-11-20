"""Small derived sidecars only; original shared CPU cache is left intact."""
import hashlib
import json
from pathlib import Path
import torch
from datasets.random_reference import RandomReference
from .io import HERE,save,sha,checkpoint
from .role_features import build

def attach(rows,role,indices,plan,halted,beat):
    dataset=RandomReference(role)
    ids=[dataset.indices[i] for i in (range(len(dataset)) if indices is None else indices)]
    identity=dict(schema='E149.public_roles.v1',base_cache_key=plan['required_cache_key'],producer_sha256=sha(HERE/'role_features.py'),
        shortest_path='Exact unweighted positive B2B/P2B graph, group separate',ports='Unique referenced coordinates; coincident ports merged by weight sum',dtype='float32 features/edges, int16 hops; no labels')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    base=HERE/'role_cache_v1'/key/dataset.role;base.mkdir(parents=True,exist_ok=True)
    meta=base.parent/'identity.json'
    if meta.exists():assert json.loads(meta.read_text(encoding='utf-8'))==identity
    else:save(meta,identity)
    groups={}
    for j,i in enumerate(ids):groups.setdefault(i//256,[]).append((j,i))
    output=[None]*len(rows);built=reused=0
    for block,entries in groups.items():
        if halted():return None
        path=base/f'{block:04d}.pt';manifest=path.with_suffix('.json');content={}
        if path.exists():
            m=json.loads(manifest.read_text(encoding='utf-8'));assert m['identity']==key and m['sha256']==sha(path)
            content=torch.load(path,map_location='cpu',weights_only=False)
        changed=False
        for j,index in entries:
            if index not in content:
                content[index]=dict(member=list(rows[j]['member']),role=build(rows[j]['graph']));built+=1;changed=True
            else:reused+=1
            assert content[index]['member']==list(rows[j]['member'])==dataset.records[index]['member']
            output[j]=dict(rows[j],role=content[index]['role'])
        if changed:
            checkpoint(path,content);save(manifest,dict(identity=key,sha256=sha(path),members={str(i):r['member'] for i,r in content.items()}))
        beat(activity='preparing/reusing public role sidecars',role_cache_built=built,role_cache_reused=reused,role_cache_loaded=built+reused,role_cache_count=len(rows))
    return output
