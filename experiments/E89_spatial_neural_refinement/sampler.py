"""Exact80/20 by slot, independently rotating complete base and new pools."""
import json,copy
from functools import lru_cache
import numpy as np
from datasets.expanded_reference import registry
from .io import HERE,ROOT,sha

@lru_cache(maxsize=1)
def manifest():
    path=ROOT/'experiments/E88_sparse_training/runs/20261031_sparse_pool/sparse_members.json'
    d=json.loads(path.read_text(encoding='utf-8'))
    for name,digest in d['identity']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Original frozen sparse preparation source changed: '+name)
    rows=d['new_members'];assert len(rows)==d['selected_count']==2268
    assert len({r['index'] for r in rows})==2268
    full=registry()['sets']['train']['records']
    for r in rows:
        assert 32000<=r['index']<64000 and r['member']==full[r['index']]['member']
    return d

def _take(state,key,count,size,seed,offset):
    part=state.setdefault(key,dict(cycle=0,cursor=0));pieces=[]
    while count:
        order=np.random.default_rng(seed+offset+100003*part['cycle']).permutation(size)
        number=min(count,size-part['cursor'])
        pieces.append(order[part['cursor']:part['cursor']+number]);count-=number;part['cursor']+=number
        if part['cursor']==size:part['cycle']+=1;part['cursor']=0
    return np.concatenate(pieces)

def epoch_order(previous,seed=0):
    state=copy.deepcopy(previous) if previous is not None else {}
    sparse=np.arange(16000)%5==4
    order=np.empty(16000,dtype=np.int64)
    order[~sparse]=_take(state,'base',12800,16000,seed,0)
    order[sparse]=16000+_take(state,'new',3200,2268,seed,10000000)
    return order,state
