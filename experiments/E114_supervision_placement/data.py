"""Raw random40k train/dev only. No legacy cache, MIB transform or test API."""
import hashlib
import json
import torch
from datasets.random_reference import RandomReference, REGISTRY
from .batches import sample, collate as batch_collate
from .ports import public_ports

_resident = {}
_checksums = None

def register_resident(pack):
    for i, row in enumerate(pack.rows): _resident[id(row)] = (pack, i)

def records(role):
    ds = RandomReference(role)
    return [ds.records[i] for i in ds.indices]

def load_rows(role, indices=None, halted=lambda: False, beat=lambda **kw: None):
    global _checksums
    ds = RandomReference(role)
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
    selected = list(range(len(ds))) if indices is None else list(indices)
    rows = []
    for offset, index in enumerate(selected):
        if halted(): return None
        record = ds.records[ds.indices[index]]
        path = ds.folder/record['payload']
        expected = _checksums[record['id']]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared payload checksum changed: '+record['payload'])
        item = ds[index]
        inputs = item['solver_inputs']
        graph = sample(inputs, torch.as_tensor(item['supervision']['reference_xywh']).float())
        graph.update(public_ports(inputs, graph['scale']))
        rows.append(dict(graph=graph, member=item['member']))
        if offset % 200 == 0: beat(activity='preparing raw CPU tensors', loaded=offset+1, count=len(selected))
    return rows

def collate(rows, device):
    if str(device).startswith('cuda') and rows and id(rows[0]) in _resident:
        pack = _resident[id(rows[0])][0]
        entries = [_resident[id(row)] for row in rows]
        if any(p is not pack for p, _ in entries): raise ValueError('Mixed resident data roles')
        return pack.batch([i for _, i in entries]), None
    from .resident import ResidentRows
    return ResidentRows(rows, device).batch(range(len(rows))), None
