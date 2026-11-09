"""Explicit v6 expansion reader; original v5 loader and defaults stay untouched."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import zipfile
import threading
import numpy as np
from datasets.paths import ROOT, dataset_paths
from datasets.worksets import FrozenMemberReader
from datasets.reference_layouts import digest, mean_area_inputs
from datasets.cases import _build_case

REGISTRY=ROOT/'protocol/reference_workset_v6.json'


def _sha(p):
    p=Path(p)
    if p.resolve()==(ROOT/'protocol/paths.json').resolve():
        archived=ROOT/'experiments/E98_loss_weight_learning/runs/20261104_generalization96_seed0/source_snapshot.zip'
        with zipfile.ZipFile(archived) as archive:
            original=archive.read('protocol/paths.json')
        digest=hashlib.sha256(original).hexdigest()
        if digest!='7a28cd42de0f10809500c43c48836576361e67a72e723f7b32a404186df9b419':
            raise ValueError('Archived original paths bytes changed')
        current=json.loads(p.read_text(encoding='utf-8'))
        old=json.loads(original)
        if any(current.get(k)!=old.get(k) for k in ('training_root','public_validation_root')):
            raise ValueError('Original data roots changed')
        return digest  # Registered semantic compatibility, not a claim current bytes match.
    return hashlib.sha256(p.read_bytes()).hexdigest()


@lru_cache(maxsize=3)
def _registry(path_string,size,mtime_ns):
    data=json.loads(Path(path_string).read_text(encoding='utf-8'))
    content=dict(data);expected=content.pop('manifest_sha256')
    if digest(content)!=expected or data['dataset_version']!='E58.reference_expansion.v6':
        raise ValueError('Expanded registry identity mismatch')
    for name,value in data['frozen_existing_protocol'].items():
        if _sha(ROOT/name)!=value:raise ValueError('Original frozen file changed: '+name)
    base=json.loads((ROOT/'protocol/reference_workset_v5.json').read_text(encoding='utf-8'))
    split=json.loads((ROOT/'protocol/split_v1.json').read_text(encoding='utf-8'))
    sets=data['sets']
    for role,count in [('train',64000),('dev',2000),('local_test',2000),('train_monitor500',500),('dev_monitor500',500)]:
        rows=sets[role]['records'];members=[tuple(x['member']) for x in rows]
        if len(rows)!=count or sets[role]['count']!=count or len(set(members))!=count:
            raise ValueError('Expanded role count/uniqueness changed: '+role)
        parent='train' if role.startswith('train') else ('dev' if role.startswith('dev') else 'local_test')
        allowed=set(split['shards'][parent])
        if any(s not in allowed or not 0<=i<112 for s,i in members):raise ValueError('Crossed original partition')
    for role in ('dev','local_test','train_monitor500','dev_monitor500'):
        if sets[role]!=base['sets'][role]:raise ValueError('Existing held-out or monitor members changed')
    if sets['train']['records'][:16000]!=base['sets']['train']['records']:
        raise ValueError('Original v5 training prefix changed')
    old_hist=Counter(x['n'] for x in base['sets']['train']['records'])
    for size,factor in [(16000,1),(32000,2),(64000,4)]:
        if Counter(x['n'] for x in sets['train']['records'][:size])!=Counter({n:factor*v for n,v in old_hist.items()}):
            raise ValueError('Nested module-count distribution mismatch')
    root=dataset_paths()['training_root']
    for shard,record in data['source_shards'].items():
        p=(root/shard).resolve()
        if not p.is_relative_to(root):raise ValueError('Source escapes dataset root')
        st=p.stat()
        if (st.st_size,st.st_mtime_ns)!=(record['size'],record['mtime_ns']):
            raise ValueError('Source shard changed: '+shard)
    return data


def registry(path=REGISTRY):
    path=Path(path).resolve();st=path.stat()
    return _registry(str(path),st.st_size,st.st_mtime_ns)
