"""Resident-shard public constructor; disk I/O separated, all construction timed.

Public extraction mirrors datasets.public_member_inputs. No free labels/specs
are indexed. A serialized shard bundles labels but only public fields are used.
"""
import time
import numpy as np
import torch
from datasets.paths import dataset_paths
from datasets.reference_layouts import registry,mean_area_inputs


class PublicShard:
    def __init__(self,role):
        if role not in ('train','dev'):raise PermissionError('Test sealed')
        self.reg=registry();self.root=dataset_paths()['training_root'];self.role=role
        self.allowed={tuple(r['member']) for r in self.reg['sets'][role]['records']}
        self.shard=None;self.raw=None

    def load(self,record):
        start=time.perf_counter();shard,row=record['member']
        if (shard,row) not in self.allowed:raise PermissionError('Outside frozen public partition')
        path=(self.root/shard).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Invalid shard path')
        stat=path.stat();expected=self.reg['source_shards'][shard]
        if (stat.st_size,stat.st_mtime_ns)!=(expected['size'],expected['mtime_ns']):raise ValueError('Frozen source changed')
        if self.shard!=shard:
            self.raw=None
            self.raw=torch.load(path,map_location='cpu',weights_only=False);self.shard=shard
        return self.raw,time.perf_counter()-start


def build_public(raw,record):
    row=record['member'][1]
    features=raw[0][row].detach().cpu().float().clone()
    areas,constraints=features[:,0],features[:,1:];n=len(areas)
    if n!=record['n']:raise ValueError('Public module count mismatch')
    locked=constraints[:,:2].bool().any(1);preplaced=constraints[:,1].bool()
    specs=torch.full((n,4),-1.,dtype=torch.float32)
    specs[locked,2:]=raw[5][row][locked,:2].float()
    specs[preplaced,:2]=raw[5][row][preplaced,2:4].float()
    ordinary=[raw[k][row].detach().cpu().float().clone() for k in (1,2,3)]
    result=mean_area_inputs((n,areas,*ordinary,constraints,specs))
    assert np.all(np.asarray(specs[~locked])==-1) and np.all(np.asarray(specs[locked&~preplaced,:2])==-1)
    return result
