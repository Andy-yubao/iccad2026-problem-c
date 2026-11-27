"""Independent read-only consumer of the verified E58 cache format."""
import json
from pathlib import Path
import torch
from datasets.expanded_reference import registry, REGISTRY
from .io import ROOT, sha
CACHE=ROOT/'experiments/E58_training_data_v6/runs/20261026_reference_v6_64k/cache'
_manifests={}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def identity():
    return {'format':'E59.readonly_E58_v6.v1','registry_sha256':sha(REGISTRY),'train_manifest_sha256':sha(CACHE/'train/manifest.json'),'dev_manifest_sha256':sha(CACHE/'dev/manifest.json')}
def verify(role):
    if role not in ('train','dev'):raise PermissionError('Test cache forbidden')
    if role in _manifests:return _manifests[role]
    run=CACHE.parent
    for name in ('preparation_status.json','results.json','cache/results.json'):
        if read(run/name)['status']!='complete':raise ValueError('E58 not complete: '+name)
    if read(run/'verification.json')['status']!='passed':raise ValueError('E58 verification failed')
    m=read(CACHE/role/'manifest.json');ident=m['identity']
    if ident['registry_sha256']!=sha(REGISTRY) or ident['paths_sha256']!=sha(ROOT/'protocol/paths.json'):raise ValueError('Registry/path identity changed')
    owner=ROOT/'experiments/E58_training_data_v6'
    for name,digest in ident['feature_sources'].items():
        if sha(owner/name)!=digest:raise ValueError('Cache producer source changed: '+name)
    for name,digest in ident['dataset_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Dataset source changed: '+name)
    if sha(Path(__file__).parent/'features.py')!=ident['feature_sources']['features.py']:raise ValueError('Feature semantics differ from prepared cache')
    records=registry()['sets'][role]['records']
    if m['members']!=[x['member'] for x in records]:raise ValueError('Cache member mismatch')
    covered=[]
    for part in m['parts']:
        st=(ROOT/part['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Cache part changed')
        covered.extend(part['indices'])
    if sorted(covered)!=list(range(len(records))):raise ValueError('Missing or duplicate cache indices')
    _manifests[role]=m
    return m
def load_rows(role,indices=None,*,train_size=64000):
    if role not in ('train','dev','train_monitor500','dev_monitor500'):raise PermissionError('Test roles sealed')
    data=registry();parent='train' if role.startswith('train') else 'dev'
    manifest=verify(parent);records=data['sets'][parent]['records']
    role_records=data['sets'][role]['records']
    if role=='train':role_records=role_records[:train_size]
    chosen=list(range(len(role_records))) if indices is None else list(indices)
    if any(i<0 or i>=len(role_records) for i in chosen):raise IndexError('Outside authorized train/dev view')
    lookup={tuple(x['member']):i for i,x in enumerate(records)}
    requested=[lookup[tuple(role_records[i]['member'])] for i in chosen];needed=set(requested);found={}
    for entry in manifest['parts']:
        if not needed.intersection(entry['indices']):continue
        part=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False,mmap=True)
        if part['identity']!=manifest['part_identities'][entry['identity_key']] or part['indices']!=entry['indices']:raise ValueError('Cache chunk identity changed')
        for index,row in zip(part['indices'],part['rows']):
            if list(row['member'])!=records[index]['member']:raise ValueError('Cached member mismatch')
            if index in needed:found[index]=row
    if set(found)!=needed:raise ValueError('Incomplete cache')
    return [found[i] for i in requested]
