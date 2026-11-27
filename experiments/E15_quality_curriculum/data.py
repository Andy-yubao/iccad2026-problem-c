"""Experiment-specific features, objectives' supervision, and padded batches."""
import torch
from datasets.quality_layouts import QualityLayouts
from .features import graph_features, encode_target


def prepare(item):
    case = item['case']
    inputs = case.solver_inputs
    f = graph_features(inputs)
    scale, n = f['scale'], case.block_count
    locked = f['locked']
    graph = {'features': f['features'], 'adjacency': f['adjacency'],
             'mib': f['relations'][0], 'group': f['relations'][1], 'mask': f['mask'],
             'known': torch.cat(((f['specs'][:, :2] + f['specs'][:, 2:] / 2) / scale,
                 (f['specs'][:, 2].clamp_min(1e-8) / f['specs'][:, 3].clamp_min(1e-8)).log()[:, None]), -1) * (1 - f['mask']),
             'area': torch.where(locked, f['specs'][:, 2:].prod(1), f['areas']) / scale.square(),
             'specs': torch.where(locked[:, None], f['specs'] / scale, torch.zeros_like(f['specs'])),
             'constraints': torch.as_tensor(inputs[5]).long(), 'scale': scale,
             'valid': torch.ones(n, dtype=torch.bool)}
    b2b, p2b = [torch.as_tensor(inputs[i]).float() for i in (2, 3)]
    b2b, p2b = [edges[~(edges == -1).all(1)] for edges in (b2b, p2b)]
    targets = {'geometry': encode_target(torch.as_tensor(item['positions']).float(), f),
               'wire_anchor': float(case.baseline_metrics['hpwl_baseline']) / float(scale),
               'area_anchor': float(case.baseline_metrics['area_baseline']) / float(scale.square()),
               'b2b': b2b, 'p2b': p2b, 'pins': torch.as_tensor(inputs[4]).float() / scale,
               'groups': [torch.where(graph['constraints'][:, 3] == value)[0] for value in
                          graph['constraints'][:, 3].unique().tolist() if value > 0],
               'mibs': [torch.where(graph['constraints'][:, 2] == value)[0] for value in
                        graph['constraints'][:, 2].unique().tolist() if value > 0]}
    return {'graph': graph, 'target': targets, 'member': item['member'], 'case': case}


def load_rows(role, indices=None, *, on_progress=None, use_cache=True):
    if use_cache:
        from .cache import PackedRows
        return PackedRows(role, indices=indices)
    dataset = QualityLayouts(role)
    ids = list(range(len(dataset))) if indices is None else list(indices)
    # Group I/O by shard without changing frozen row/order or the training shuffle.
    rows = [None] * len(ids)
    for count, slot in enumerate(sorted(range(len(ids)), key=lambda i: dataset.records[ids[i]]['member'])):
        rows[slot] = prepare(dataset[ids[slot]])
        if on_progress is not None and (count + 1) % 100 == 0:
            on_progress()
        if (count + 1) % 1000 == 0:
            print(f'Loaded {role}: {count + 1}/{len(ids)}', flush=True)
    return rows


def collate(rows, device):
    size = max(len(r['graph']['valid']) for r in rows)
    pair_keys = {'adjacency', 'mib', 'group'}
    graph = {}
    for key, value in rows[0]['graph'].items():
        shape = (len(rows),) + ((size, size) if key in pair_keys else
                (() if value.ndim == 0 else (size,) + value.shape[1:]))
        tensor = torch.zeros(shape, dtype=value.dtype)
        for i, row in enumerate(rows):
            item = row['graph'][key]
            if key in pair_keys:
                tensor[i, :len(item), :len(item)] = item
            elif item.ndim == 0:
                tensor[i] = item
            else:
                tensor[i, :len(item)] = item
        graph[key] = tensor.to(device)
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else
               [x.to(device) for x in v] if isinstance(v, list) else v
               for k, v in row['target'].items()} for row in rows]
    return graph, targets
