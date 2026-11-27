"""Reuse original checked tensors, then attach only versioned public-role extras."""
import json
from datasets.random_reference16k import REGISTRY, subset_registry
from .base_data import cache_identity,make_row,register_resident,collate
from .base_data import records as parent_records
from .base_data import load_rows as base_load
from .role_cache import attach
from .io import HERE
def load_rows(role,indices=None,halted=lambda:False,beat=lambda **kw:None):
    rows=base_load(role,indices,halted,beat)
    if rows is None:return None
    return attach(rows,role,indices,json.loads((HERE/'plan.json').read_text(encoding='utf-8')),halted,beat)

def training_indices():
    return subset_registry()[0]['train_parent_indices']

def records(role):
    if role == 'dev_monitor500':
        return parent_records(role)
    subset, parent = subset_registry()
    entries = parent['sets']['train']['records']
    selected = subset['train_parent_indices']
    local = subset['panels']['train_monitor500'] if role == 'train_monitor500' else range(len(selected))
    if role not in ('train_monitor500', 'train'):
        raise ValueError(role)
    return [entries[selected[i]] for i in local]
