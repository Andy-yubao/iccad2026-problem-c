"""Read frozen members' public inputs without indexing free geometry or scores."""
import numpy as np
import torch
from .paths import dataset_paths
from .reference_layouts import registry, mean_area_inputs


def public_member_inputs(record, role, *, authorize_local_test=False):
    if role not in ('train', 'dev', 'local_test'):
        raise ValueError(role)
    if role == 'local_test' and not authorize_local_test:
        raise PermissionError('Explicit local-test public-input inference authorization required')
    reg = registry()
    if record not in reg['sets'][role]['records']:
        raise ValueError('Member is outside the requested frozen partition')
    shard, row = record['member']
    root = dataset_paths()['training_root']
    path = (root / shard).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Invalid shard path')
    source = reg['source_shards'][shard]
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source['size'], source['mtime_ns']):
        raise ValueError('Frozen source changed')
    # The serialized container bundles labels. Only public tensor fields and
    # publicly specified locked WH / preplaced XY are indexed below.
    raw = torch.load(path, map_location='cpu', weights_only=False)
    features = raw[0][row].detach().cpu().float().clone()
    areas, constraints = features[:, 0], features[:, 1:]
    n = len(areas)
    assert n == record['n']
    locked = constraints[:, :2].bool().any(1)
    preplaced = constraints[:, 1].bool()
    specs = torch.full((n, 4), -1., dtype=torch.float32)
    specs[locked, 2:] = raw[5][row][locked, :2].float()
    specs[preplaced, :2] = raw[5][row][preplaced, 2:4].float()
    ordinary = [raw[k][row].detach().cpu().float().clone() for k in (1, 2, 3)]
    inputs = mean_area_inputs((n, areas, *ordinary, constraints, specs))
    assert np.all(np.asarray(specs[~locked]) == -1)
    assert np.all(np.asarray(specs[locked & ~preplaced, :2]) == -1)
    return inputs
