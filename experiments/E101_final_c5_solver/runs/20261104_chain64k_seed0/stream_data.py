"""Independent exact-FP32 memmap adapter for the original E42/E74 trainers.

Installed as each stage's own data.py. Dynamic E42 padding and fixed E74
padding are preserved. Only one batch is transferred; the dataset stays on CPU.
"""
from pathlib import Path
import json
import numpy as np
import torch
from .io import ROOT,HERE,sha
from .dataset_view import ReferenceLayouts,registry,REGISTRY
from .batches import sample,collate as batch_collate
from .cache import verify,load_rows as original_rows

NODE=('features','mask','area','known','target','specs','reference','constraints')
PAIR=('adjacency','mib','group')
CACHE=HERE/'runs/prepared64k_memmap'
_sets={}

def check_arrays(parent):
    manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['registry_sha256']==sha(REGISTRY)
    for name,e in manifest['arrays'][parent].items():
        s=(CACHE/parent/name).stat();assert (s.st_size,s.st_mtime_ns)==(e['size'],e['mtime_ns']),name

class Graph:
    def __init__(self,owner,index):self.owner,self.index=owner,index
    def __getitem__(self,key):
        a=self.owner.arrays;j=int(self.owner.indices[self.index]);n=int(a['n'][j])
        if key=='n':return n
        if key=='scale':return torch.tensor(float(a[key][j]),dtype=torch.float32)
        return torch.from_numpy(a[key][j,:n,:n].copy() if key in PAIR else a[key][j,:n].copy())

class Row:
    def __init__(self,owner,index):self.owner,self.index=owner,index;self.graph=Graph(owner,index)
    def __getitem__(self,key):
        if key=='graph':return self.graph
        if key=='member':return tuple(self.owner.members[self.index])
        if key=='case':return self.owner.case(self.index)
        raise KeyError(key)

class Rows:
    def __init__(self,role):
        self.role=role;self.parent='train' if role.startswith('train') else 'dev'
        verify(self.parent);check_arrays(self.parent)
        sets=registry()['sets'];full=sets[self.parent]['records'];records=sets[role]['records']
        self.members=[r['member'] for r in records];lookup={tuple(r['member']):i for i,r in enumerate(full)}
        self.indices=np.array([lookup[tuple(r['member'])] for r in records],dtype=np.int64)
        self.arrays={k:np.load(CACHE/self.parent/(k+'.npy'),mmap_mode='r') for k in (*NODE,*PAIR,'scale','n')}
        self.rows=[Row(self,i) for i in range(len(records))];self.pack=None;self.cases=None;self.host={}
    def __len__(self):return len(self.rows)
    def __iter__(self):return iter(self.rows)
    def __getitem__(self,index):return self.rows[index]
    def case(self,index):
        if self.cases is None:
            # Original cached cases are needed only for the frozen 500-member
            # diagnostics; they never enter the network input packer.
            role='train_monitor500' if self.parent=='train' else 'dev_monitor500'
            self.cases={tuple(r['member']):r['case'] for r in original_rows(role)}
        return self.cases[tuple(self.members[index])]
    def batch(self,indices,device,fixed=False,out=None):
        if isinstance(indices,torch.Tensor):indices=indices.detach().cpu().numpy()
        ids=self.indices[np.asarray(indices,dtype=np.int64)];count=len(ids);assert 0<count<=32
        n=120 if fixed else int(self.arrays['n'][ids].max());values={}
        for key in (*NODE,*PAIR,'scale','valid'):
            if key=='valid':a=np.arange(n)[None,:]<self.arrays['n'][ids,None]
            elif key=='scale':a=np.take(self.arrays[key],ids,axis=0)
            else:
                a=np.take(self.arrays[key],ids,axis=0)
                a=a[:,:n,:n] if key in PAIR else a[:,:n]
            a=np.ascontiguousarray(a);t=torch.from_numpy(a)
            if str(device).startswith('cuda'):
                if key not in self.host or self.host[key].shape!=t.shape:self.host[key]=torch.empty_like(t,pin_memory=True)
                self.host[key].copy_(t);t=self.host[key]
            if out is None:values[key]=t.to(device,non_blocking=True)
            else:out[key].copy_(t,non_blocking=True)
        return values if out is None else out

def load_rows(role,indices=None,*,use_cache=True):
    if role not in ('train','dev','train_monitor500','dev_monitor500'):raise PermissionError('Test payloads sealed')
    if not use_cache:raise ValueError('This run uses the registered exact frozen disk cache')
    if role not in _sets:_sets[role]=Rows(role)
    rows=_sets[role]
    return rows if indices is None else [rows[i] for i in indices]

def register_resident(pack):pack.rows.pack=pack

def collate(rows,device):
    assert rows and all(r.owner is rows[0].owner for r in rows)
    owner=rows[0].owner
    return owner.batch([r.index for r in rows],device,fixed=owner.pack is not None),None
