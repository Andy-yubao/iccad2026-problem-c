"""Read only the newly prepared CPU rows; fixed old MIB32k/dev500 membership."""
import json
import torch
from .io import ROOT,HERE,sha
CACHE_OWNER=ROOT/'experiments/E153_mib_role_topology'
REGISTRY=ROOT/'protocol/reference_workset_v6.json'
_resident={};_loaded={}
def records(role):
    registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
    return registry['sets'][role]['records'][:32000] if role=='train' else registry['sets'][role]['records']
def load_rows(role,indices=None,halted=lambda:False,beat=lambda **kw:None):
    if role not in ('train','train_monitor500','dev_monitor500'):raise PermissionError('Only registered train/dev panels')
    cache=CACHE_OWNER/'data_cache_v1';status=json.loads((CACHE_OWNER/'data_preparation_state.json').read_text(encoding='utf-8'))
    if status['status']!='complete':raise RuntimeError('Data preparation is not complete')
    parent='train' if role.startswith('train') else 'dev';m=json.loads((cache/parent/'manifest.json').read_text(encoding='utf-8'))
    selected=records(role);selected=selected if indices is None else [selected[i] for i in indices]
    wanted={tuple(r['member']) for r in selected};lookup={tuple(v):i for i,v in enumerate(m['members'])};needed={lookup[v] for v in wanted};found={}
    for part in m['parts']:
        if not needed.intersection(part['indices']):continue
        if halted():return None
        path=CACHE_OWNER/part['path']
        if sha(path)!=part['sha256']:raise ValueError('Prepared chunk changed')
        content=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert content['identity']==status['identity'] and content['indices']==part['indices']
        for i,row in zip(content['indices'],content['rows']):
            if i in needed:
                assert list(row['member'])==m['members'][i];found[tuple(row['member'])]=row
        beat(activity='loading prepared MIB/role CPU cache',loaded=len(found),count=len(selected))
    if set(found)!=wanted:raise ValueError('Missing registered members')
    return [found[tuple(r['member'])] for r in selected]
def register_resident(pack):
    for i,row in enumerate(pack.rows):_resident[id(row)]=(pack,i)
def collate(rows,device):
    if str(device).startswith('cuda') and id(rows[0]) in _resident:
        pack=_resident[id(rows[0])][0];return pack.batch([_resident[id(row)][1] for row in rows]),None
    from .resident import ResidentRows
    return ResidentRows(rows,device,max_nodes=120).batch(range(len(rows))),None
