"""Read-only verified E58 tensors, with explicit archive-path migration."""
import json
from pathlib import Path
import torch
from datasets.expanded_reference import registry,REGISTRY
from .io import ROOT,HERE,sha
OWNER=ROOT/'history/20261029_model_research_consolidation/experiments/E58_training_data_v6'
CACHE=OWNER/'runs/20261026_reference_v6_64k/cache'
OLD='experiments/E58_training_data_v6/'
_manifests={}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def resolve_part(name):
    name=name.replace(chr(92),'/')
    if name.startswith(OLD):target=(OWNER/name[len(OLD):]).resolve()
    elif name.startswith('experiments/E42_joint_regularization/runs/prepared_v5_e10_cache_v1/'):
        target=(ROOT/name).resolve()
    else:raise ValueError('Unregistered cache root: '+name)
    if not target.is_relative_to(ROOT):raise ValueError('Cache path escapes workspace')
    return target

def identity():return {'format':'E91.readonly_archived_E58.v6prefix32k.v1','registry_sha256':sha(REGISTRY),'train_manifest_sha256':sha(CACHE/'train/manifest.json'),'dev_manifest_sha256':sha(CACHE/'dev/manifest.json'),'train_size':32000,'old_cache_owner':OLD,'resolved_cache_owner':OWNER.relative_to(ROOT).as_posix()}

def verify(role):
    if role not in ('train','dev'):raise PermissionError('Test caches sealed')
    if role in _manifests:return _manifests[role]
    for name in ('preparation_status.json','results.json','cache/results.json'):
        if read(CACHE.parent/name)['status']!='complete':raise ValueError('E58 producer incomplete')
    if read(CACHE.parent/'verification.json')['status']!='passed':raise ValueError('E58 verification failed')
    m=read(CACHE/role/'manifest.json');ident=m['identity']
    if ident['registry_sha256']!=sha(REGISTRY) or ident['paths_sha256']!=sha(ROOT/'protocol/paths.json'):raise ValueError('Registry/path drift')
    for name,digest in ident['feature_sources'].items():
        if sha(OWNER/name)!=digest:raise ValueError('Archived producer source drift: '+name)
    for name,digest in ident['dataset_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Dataset source drift: '+name)
    for name in ('features.py','batches.py'):
        if sha(HERE/name)!=ident['feature_sources'][name]:raise ValueError('Input semantics differ: '+name)
    members=[x['member'] for x in registry()['sets'][role]['records']]
    if members!=m['members']:raise ValueError('Frozen member mismatch')
    covered=[]
    for part in m['parts']:
        st=resolve_part(part['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Cache part drift')
        covered.extend(part['indices'])
    if sorted(covered)!=list(range(len(members))):raise ValueError('Cache coverage mismatch')
    _manifests[role]=m;return m

def load_rows(role,indices=None):
    if role not in ('train','dev','train_monitor500','dev_monitor500'):raise PermissionError('Test payloads sealed')
    data=registry();parent='train' if role.startswith('train') else 'dev';m=verify(parent)
    full=data['sets'][parent]['records'];role_records=data['sets'][role]['records']
    if role=='train':role_records=role_records[:32000]
    chosen=list(range(len(role_records))) if indices is None else list(indices)
    if any(i<0 or i>=len(role_records) for i in chosen):raise IndexError('Outside frozen view')
    lookup={tuple(x['member']):i for i,x in enumerate(full)}
    requested=[lookup[tuple(role_records[i]['member'])] for i in chosen];needed=set(requested);found={}
    for entry in m['parts']:
        if not needed.intersection(entry['indices']):continue
        part=torch.load(resolve_part(entry['path']),map_location='cpu',weights_only=False,mmap=True)
        if part['identity']!=m['part_identities'][entry['identity_key']] or part['indices']!=entry['indices']:raise ValueError('Cache chunk identity drift')
        for index,row in zip(part['indices'],part['rows']):
            if list(row['member'])!=full[index]['member']:raise ValueError('Cached member mismatch')
            if index in needed:found[index]=row
    if set(found)!=needed:raise ValueError('Missing requested cache rows')
    print('Verified archived cache loaded:',role,len(requested),'cases',flush=True)
    return [found[i] for i in requested]
