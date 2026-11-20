"""Frozen fresh MIB train/dev caches. Test free labels remain inaccessible."""
import json
import pathlib
import hashlib
import torch

HERE=pathlib.Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for blob in iter(lambda:f.read(4*2**20),b''):
            h.update(blob)
    return h.hexdigest()


def load_rows(role,indices=None,beat=lambda **kw:None):
    if role not in ('train','dev','train_monitor500','dev_monitor500'):
        raise PermissionError('Only new train/dev cache access is authorized')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    if read(HERE/'state.json')['status']!='complete':
        raise RuntimeError('Dataset preparation is not complete')
    registry=read(HERE/'manifest.json')
    parent=role.split('_')[0]
    selected=registry['panels'][role] if 'monitor' in role else list(range(registry['sets'][parent]['count']))
    if indices is not None:selected=[selected[i] for i in indices]
    wanted=set(selected);found={}
    manifest=read(HERE/f'cache/{parent}/manifest.json')
    assert manifest['identity']==registry['manifest_sha256']
    for part in manifest['parts']:
        if not wanted.intersection(part['indices']):continue
        path=HERE/part['path']
        assert sha(path)==part['sha256']
        content=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert content['identity']==manifest['identity'] and content['indices']==part['indices']
        for i,row in zip(content['indices'],content['rows']):
            if i in wanted:
                assert list(row['member'])==manifest['members'][i]
                found[i]=row
        beat(loaded=len(found),count=len(selected))
    assert set(found)==wanted
    return [found[i] for i in selected]
