"""Read frozen nested memberships; never inspect labels to select cases."""
import json
from .io import HERE, ROOT, sha


def indices(size, seed):
    data = json.loads((HERE/'subsets.json').read_text(encoding='utf-8'))
    if sha(ROOT/'protocol/reference_workset_v5.json') != data['registry_sha256']:
        raise ValueError('Registry changed')
    if size == 32:
        return data['memory32_indices']
    values = data['families'][str(seed)][str(size)]
    assert len(values) == size and len(set(values)) == size
    assert set(data['mandatory_train_monitor_indices']) <= set(values)
    return values
