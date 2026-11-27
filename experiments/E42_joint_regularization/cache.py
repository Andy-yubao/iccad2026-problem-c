"""Read-only reuse of the frozen E25/E17 train/dev tensor cache."""
import json
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from .io import ROOT,HERE,sha

CACHE=HERE/'runs/prepared_v5_e10_cache_v1'

def identity():
    return json.loads((CACHE/'train/manifest.json').read_text())['identity']

def verify(role):
    if role not in ('train','dev'):raise PermissionError('Tests remain sealed')
    manifest=json.loads((CACHE/role/'manifest.json').read_text());recorded=manifest['identity']
    if recorded!=identity() or recorded['registry']!=sha(REGISTRY) or recorded['paths']!=sha(ROOT/'protocol/paths.json'):raise ValueError('Cache registry/path identity changed')
    for name in ('features.py','batches.py'):
        if sha(HERE/name)!=recorded['sources'][name]:raise ValueError('Tensor semantics changed: '+name)
    for name,digest in recorded['datasets_sources'].items():
        if sha(ROOT/'datasets'/name)!=digest:raise ValueError('Dataset source changed: '+name)
    if manifest['members']!=[list(r['member']) for r in ReferenceLayouts(role).records]:raise ValueError('Cache members changed')
    for part in manifest['parts']:
        st=(CACHE/role/part['name']).stat()
        if (st.st_size,st.st_mtime_ns)!=(part['size'],part['mtime_ns']):raise ValueError('Cache part changed')
    return manifest
