"""Original frozen tensors; only the archived E42 path is migrated in this process."""
from .. import cache
from ..data import load_rows, collate, register_resident
from ..io import ROOT

def resolve_part(name):
    name = name.replace('\\','/')
    if name.startswith(cache.OLD):
        path = cache.OWNER/name[len(cache.OLD):]
    elif name.startswith('experiments/E42_joint_regularization/runs/prepared_v5_e10_cache_v1/'):
        path = ROOT/'history/20261103_final_delivery_consolidation'/name
    else:
        raise ValueError('Unregistered cache root: '+name)
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Cache escapes workspace')
    return path

cache.resolve_part = resolve_part
