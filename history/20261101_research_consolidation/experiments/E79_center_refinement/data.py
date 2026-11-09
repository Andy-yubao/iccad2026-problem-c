"""Read-only indexed E17 cache reuse; free reference geometry is supervision."""
import torch
from datasets.reference_layouts import public_hard_reference
from .dataset_view import ReferenceLayouts
from .batches import sample, collate as batch_collate

_loaded = {}
_resident = {}


def register_resident(pack):
    for i, row in enumerate(pack.rows): _resident[id(row)] = (pack, i)


def prepare(item):
    case = item['case']
    geometry = sample(case.solver_inputs, torch.as_tensor(item['positions']).float())
    case.reference_positions = public_hard_reference(
        case.solver_inputs[1], case.solver_inputs[5], case.solver_inputs[6])
    return {'graph': geometry, 'member': item['member'], 'case': case}


def collate(rows, device):
    if str(device).startswith('cuda') and rows and id(rows[0]) in _resident:
        pack = _resident[id(rows[0])][0]
        entries = [_resident[id(row)] for row in rows]
        if any(p is not pack for p, _ in entries): raise ValueError('Cannot mix resident data roles')
        return pack.batch([i for _, i in entries]), None
    return batch_collate([row['graph'] for row in rows], list(range(len(rows))), device), None


def load_rows(role, indices=None, *, use_cache=True):
    dataset=ReferenceLayouts(role)
    selected=list(range(len(dataset))) if indices is None else list(indices)
    if use_cache:
        from .cache import load_rows as cached
        return cached(role,selected)
    return [prepare(dataset[i]) for i in selected]
