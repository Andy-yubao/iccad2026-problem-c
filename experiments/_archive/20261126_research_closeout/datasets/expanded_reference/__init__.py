"""Explicit v6 expansion reader; original v5 loader and defaults stay untouched."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import threading
import numpy as np
from datasets.paths import ROOT, dataset_paths
from datasets.worksets import FrozenMemberReader
from datasets.reference_layouts import digest, mean_area_inputs
from datasets.cases import _build_case

REGISTRY=ROOT/'protocol/reference_workset_v6.json'


def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


class ExpandedReferenceLayouts:
    def __init__(self,role,*,registry_path=REGISTRY,train_size=64000):
        if role not in ('train','dev','train_monitor500','dev_monitor500'):
            raise PermissionError('Test payloads and test caches remain sealed')
        if train_size not in (16000,32000,64000):raise ValueError('Unknown frozen training view')
        self.registry=registry(registry_path);self.role=role
        rows=self.registry['sets'][role]['records']
        self.records=rows[:train_size] if role=='train' else rows
        self.parent='train' if role.startswith('train') else 'dev'
        self._local=threading.local()

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        if not hasattr(self._local,'reader'):
            self._local.reader=FrozenMemberReader([x['member'] for x in self.records],self.parent,
                identity=self.registry['manifest_sha256'])
        raw=self._local.reader.case(index)
        if raw.block_count!=self.records[index]['n']:raise ValueError('Actual public module count changed')
        original=raw.solver_inputs
        inputs=mean_area_inputs(original)
        areas=np.asarray(original[1]);adjusted=np.asarray(inputs[1]);c=np.asarray(inputs[5]);s=np.asarray(inputs[6])
        groups=changed=blocks=0;max_total_relative_error=0.
        for group in sorted(set(c[:,2])-{0}):
            ids=np.flatnonzero(c[:,2]==group)
            if len(ids)<2:continue
            groups+=1;blocks+=len(ids);changed+=int(not np.array_equal(areas[ids],adjusted[ids]))
            if not np.all(adjusted[ids]==adjusted[ids[0]]):raise ValueError('MIB input areas not equal')
            error=abs(float(adjusted[ids].astype(float).sum()-areas[ids].astype(float).sum()))/max(float(areas[ids].astype(float).sum()),1e-12)
            max_total_relative_error=max(max_total_relative_error,error)
            if error>1e-6:raise ValueError('MIB total-area preservation failed')
        if not np.array_equal(np.asarray(original[6]),s):raise ValueError('Locked public specs changed')
        case=_build_case(raw.case_id,raw.source,original[1:6],raw.reference_positions,raw.stored_metrics)
        case.solver_inputs=inputs
        return {'member':tuple(self.records[index]['member']),'case':case,
                'positions':np.asarray(raw.reference_positions,dtype=np.float32),
                'mib_audit':{'nontrivial_groups':groups,'changed_groups':changed,'grouped_blocks':blocks,
                    'max_group_total_relative_error':max_total_relative_error,'locked_specs_unchanged':True}}
