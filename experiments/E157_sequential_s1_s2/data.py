"""Shared fresh shape-corrected MIB36k CPU tensors; no test loader."""
import json
import torch
from .io import ROOT,HERE,sha
CACHE_OWNER=ROOT/'datasets/random_mib40k_v1'
REGISTRY=ROOT/'protocol/random_mib40k_v1.json'
_resident={}

def records(role):
    if role not in ('train','train_monitor500','dev_monitor500'):
        raise PermissionError('Only registered train and monitoring roles')
    registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
    parent=role.split('_')[0];rows=registry['sets'][parent]['records']
    return [rows[i] for i in registry['panels'][role]] if 'monitor' in role else rows

def load_rows(role,indices=None,halted=lambda:False,beat=lambda **kw:None):
    selected=records(role)
    if indices is not None:selected=[selected[i] for i in indices]
    status=json.loads((CACHE_OWNER/'state.json').read_text(encoding='utf-8'))
    registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
    assert status['status']=='complete' and status['identity']==registry['manifest_sha256']
    parent=role.split('_')[0]
    manifest=json.loads((CACHE_OWNER/f'cache/{parent}/manifest.json').read_text(encoding='utf-8'))
    assert manifest['identity']==status['identity']
    wanted={tuple(r['member']) for r in selected}
    lookup={tuple(v):i for i,v in enumerate(manifest['members'])}
    needed={lookup[v] for v in wanted};found={}
    for part in manifest['parts']:
        if not needed.intersection(part['indices']):continue
        if halted():return None
        path=CACHE_OWNER/part['path']
        assert sha(path)==part['sha256'],'Prepared cache changed'
        content=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert content['identity']==status['identity'] and content['indices']==part['indices']
        for i,row in zip(content['indices'],content['rows']):
            if i in needed:
                assert list(row['member'])==manifest['members'][i]
                found[tuple(row['member'])]=row
        beat(activity='loading frozen fresh MIB/shape-corrected CPU cache',loaded=len(found),count=len(selected))
    assert set(found)==wanted
    return [found[tuple(r['member'])] for r in selected]

def register_resident(pack):
    for i,row in enumerate(pack.rows):_resident[id(row)]=(pack,i)

def collate(rows,device):
    if str(device).startswith('cuda') and id(rows[0]) in _resident:
        pack=_resident[id(rows[0])][0]
        return pack.batch([_resident[id(row)][1] for row in rows]),None
    from .resident import ResidentRows
    return ResidentRows(rows,device,max_nodes=120).batch(range(len(rows))),None
