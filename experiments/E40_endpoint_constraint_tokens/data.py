"""Read-only indexed E17 cache reuse; free reference geometry is supervision."""
import torch
from datasets.reference_layouts import ReferenceLayouts, public_hard_reference
from .batches import sample, collate as batch_collate

_loaded = {}


def prepare(item):
    case = item['case']
    geometry = sample(case.solver_inputs, torch.as_tensor(item['positions']).float())
    case.reference_positions = public_hard_reference(
        case.solver_inputs[1], case.solver_inputs[5], case.solver_inputs[6])
    return {'graph': geometry, 'member': item['member'], 'case': case}


def collate(rows, device):
    from .public import augment
    batch = batch_collate([row['graph'] for row in rows], list(range(len(rows))), device)
    batch = augment(batch, [row['case'].solver_inputs for row in rows],
                    [row['graph']['scale'] for row in rows], device)
    return batch, None


def load_rows(role, indices=None, *, use_cache=True):
    dataset = ReferenceLayouts(role)  # Reject sealed roles before reading payloads.
    selected = list(range(len(dataset))) if indices is None else list(indices)
    if not use_cache:
        return [prepare(dataset[i]) for i in selected]
    from .cache import CACHE, identity, verify
    parent = 'train' if role.startswith('train') else 'dev'
    if parent not in _loaded:
        manifest = verify(parent)
        rows = [None] * manifest['count']
        for entry in manifest['parts']:
            part = torch.load(CACHE/parent/entry['name'], map_location='cpu', weights_only=False)
            if part['identity'] != identity() or len(part['indices']) != len(part['rows']):
                raise ValueError('Prepared chunk identity/count mismatch')
            for index, row in zip(part['indices'], part['rows']):
                if not 0 <= index < len(rows) or rows[index] is not None:
                    raise ValueError('Duplicate or out-of-range prepared index')
                if list(row['member']) != manifest['members'][index]:
                    raise ValueError('Prepared logical index/member mismatch')
                rows[index] = row
        if any(row is None for row in rows):
            raise ValueError('Incomplete prepared data')
        _loaded[parent] = {tuple(row['member']): row for row in rows}
        print(f'Prepared tensor cache loaded: {parent}, {len(rows)} cases', flush=True)
    return [_loaded[parent][tuple(dataset.records[i]['member'])] for i in selected]
