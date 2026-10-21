"""Exact registered reuse of frozen E17 input/reference tensor artifacts."""
import json
from datasets.reference_layouts import ReferenceLayouts,REGISTRY
from .io import ROOT,HERE,sha

CACHE=ROOT/'experiments/E25_integrated_baseline/runs/prepared_v5_e10_cache_v1'


def identity():
    registered=json.loads((HERE/'provenance.json').read_text())['prepared_cache_identity']
    if registered['registry']!=sha(REGISTRY) or registered['paths']!=sha(ROOT/'protocol/paths.json'):
        raise ValueError('Frozen artifact data identity changed')
    if any(sha(ROOT/'datasets'/name)!=digest for name,digest in registered['datasets_sources'].items()):
        raise ValueError('Shared data reader changed since preparation')
    return registered


def verify(role):
    if role not in ('train','dev'):raise PermissionError('Tests remain sealed')
    folder=CACHE/role;manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['identity']!=identity():raise ValueError('Unregistered artifact identity')
    if manifest['members']!=[list(row['member']) for row in ReferenceLayouts(role).records]:
        raise ValueError('Prepared members changed')
    for entry in manifest['parts']:
        stat=(folder/entry['name']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(entry['size'],entry['mtime_ns']):raise ValueError('Prepared tensor changed')
    return manifest


if __name__=='__main__':
    for role in ['train','dev']:verify(role);print('Exact read-only cache ready:',role,flush=True)
