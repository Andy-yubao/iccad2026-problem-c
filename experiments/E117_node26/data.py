"""Checked random40k train/dev tensors with shared persistent static preprocessing."""
import hashlib
import json
import torch
from datasets.random_reference import RandomReference, REGISTRY
from .batches import sample, collate as batch_collate
from .ports import public_ports
from datasets.static_tensor_cache import load_rows as cached_rows, sha
from pathlib import Path

_resident = {}
_checksums = None

def register_resident(pack):
    for i, row in enumerate(pack.rows): _resident[id(row)] = (pack, i)

def records(role):
    ds = RandomReference(role)
    return [ds.records[i] for i in ds.indices]

def make_row(inputs, reference, member):
    graph=sample(inputs,reference)
    graph.update(public_ports(inputs,graph['scale']))
    return dict(graph=graph,member=member)

def cache_identity(plan):
    here=Path(__file__).parent
    return dict(schema='static-original-no-mib-all-predictions.v1',
        parent_registry_sha256=sha(REGISTRY),
        dataset_report_sha256=plan['dataset_report_sha256'],
        payload_checksums_sha256=plan['payload_checksums_sha256'],
        producers={name:sha(here/name) for name in ('features.py','batches.py','ports.py')},
        precision='CPU float32; integers/bools retained',torch=str(torch.__version__),
        policy='original area; no MIB; predict all; labels outside public input; no augmentation; no model state')

def load_rows(role, indices=None, halted=lambda: False, beat=lambda **kw: None):
    global _checksums
    ds = RandomReference(role)
    from .io import OWNER
    plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
    if _checksums is None:
        from .io import OWNER
        plan = json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        blob = (ds.folder/'checksums.jsonl').read_bytes()
        report = json.loads((ds.folder/'report.json').read_text(encoding='utf-8'))
        if (hashlib.sha256(blob).hexdigest() != plan['payload_checksums_sha256'] or
                report['checksums_sha256'] != plan['payload_checksums_sha256'] or
                hashlib.sha256((ds.folder/'report.json').read_bytes()).hexdigest() != plan['dataset_report_sha256']):
            raise ValueError('Dataset checksum manifest changed')
        _checksums = {}
        for line in blob.decode('utf-8').splitlines():
            entry = json.loads(line)
            for record in entry['records']:
                _checksums[record['id']] = record['payload_sha256']
    return cached_rows(ds,__name__,cache_identity(plan),_checksums,indices,halted,beat,
                       workers=plan.get('cache_build_workers',4))

def collate(rows, device):
    if str(device).startswith('cuda') and rows and id(rows[0]) in _resident:
        pack = _resident[id(rows[0])][0]
        entries = [_resident[id(row)] for row in rows]
        if any(p is not pack for p, _ in entries): raise ValueError('Mixed resident data roles')
        return pack.batch([i for _, i in entries]), None
    from .resident import ResidentRows
    return ResidentRows(rows, device).batch(range(len(rows))), None
