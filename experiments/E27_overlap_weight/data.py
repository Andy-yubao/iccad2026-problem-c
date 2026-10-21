"""Original reference rectangles remain supervision; public inputs stay masked."""
import torch
from datasets.reference_layouts import ReferenceLayouts, public_hard_reference
from .batches import sample, collate as batch_collate

_loaded = {}


def prepare(item):
    case = item['case']
    geometry = sample(case.solver_inputs, torch.as_tensor(item['positions']).float())
    # The scorer only needs public hard specifications, never free labels.
    case.reference_positions = public_hard_reference(case.solver_inputs[1],
        case.solver_inputs[5], case.solver_inputs[6])
    return {'graph': geometry, 'member': item['member'], 'case': case}


def collate(rows, device):
    return batch_collate([row['graph'] for row in rows], list(range(len(rows))), device), None


def load_rows(role, indices=None, *, use_cache=True):
    dataset = ReferenceLayouts(role)  # Reject test roles before any payload read.
    selected = list(range(len(dataset))) if indices is None else list(indices)
    if not use_cache:
        return [prepare(dataset[i]) for i in selected]
    from .cache import CACHE, identity, verify
    parent = 'train' if role.startswith('train') else 'dev'
    key = (parent, str(CACHE))
    if key not in _loaded:
        manifest = verify(parent)
        rows = [None] * manifest['count']
        for entry in manifest['parts']:
            part = torch.load(CACHE/parent/entry['name'], map_location='cpu', weights_only=False)
            if part['identity'] != identity():
                raise ValueError('Prepared chunk identity differs')
            for index, row in zip(part['indices'], part['rows']): rows[index] = row
        if any(row is None for row in rows): raise ValueError('Incomplete prepared data')
        _loaded[key] = {tuple(row['member']): row for row in rows}
        print(f'Prepared tensor cache loaded: {parent}, {len(rows)} cases', flush=True)
    return [_loaded[key][tuple(dataset.records[i]['member'])] for i in selected]
