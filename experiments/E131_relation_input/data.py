"""Read-only frozen16k views of the existing complete parent CPU cache."""
import functools
import json
from pathlib import Path
import torch
from datasets.random_reference16k import subset_registry
from .control import ROOT, OWNER, read, sha
from .resident import ResidentRows

REGISTRY = ROOT / 'protocol/random_reference16k_v1.json'
ROLES = ('train', 'train_monitor500', 'dev_monitor500')
_resident = {}


@functools.lru_cache(maxsize=1)
def registries():
    plan = read(OWNER / 'plan.json')
    if sha(REGISTRY) != plan['subset_registry_sha256']:
        raise ValueError('Frozen16k registry changed')
    if sha(ROOT / plan['parent_registry']) != plan['parent_registry_sha256']:
        raise ValueError('Parent registry changed')
    return subset_registry(REGISTRY)


def role_indices(role):
    if role not in ROLES:
        raise PermissionError('Only frozen train16k/train500/dev500 cache views are allowed; test is sealed')
    sub, _ = registries()
    if role == 'dev_monitor500':
        return 'dev', sub['panels']['dev_monitor500']
    ids = sub['train_parent_indices']
    if role == 'train_monitor500':
        ids = [ids[i] for i in sub['panels']['train_monitor500']]
    return 'train', ids


def records(role):
    parent_role, indices = role_indices(role)
    _, parent = registries()
    return [parent['sets'][parent_role]['records'][i] for i in indices]


@functools.lru_cache(maxsize=1)
def cache_folder():
    plan = read(OWNER / 'plan.json')
    path = ROOT / 'datasets/static_tensor_cache_v1' / plan['required_cache_key']
    identity = path / 'identity.json'
    if not path.is_dir() or sha(identity) != plan['cache_identity_sha256']:
        raise ValueError('Required existing CPU cache is missing or changed; rebuild is disabled')
    value = read(identity)
    key = __import__('hashlib').sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if key != plan['required_cache_key'] or value['schema'] != 'static-original-no-mib-all-predictions.v1':
        raise ValueError('Incompatible CPU cache identity')
    if (value['dataset_report_sha256'] != plan['dataset_report_sha256']
            or value['payload_checksums_sha256'] != plan['payload_checksums_sha256']
            or value['parent_registry_sha256'] != plan['parent_registry_sha256']
            or value['torch'] != str(torch.__version__)):
        raise ValueError('CPU cache provenance/dtype/producer environment changed')
    return path


def cache_inventory():
    """Verify complete membership from sidecars, without opening dev tensors."""
    result = {}
    for role in ('train', 'dev_monitor500'):
        parent_role, ids = role_indices(role)
        groups = {}
        for index in ids:
            groups.setdefault(index // 256, []).append(index)
        recs = records(role)
        lookup = dict(zip(ids, recs))
        for block, selected in groups.items():
            path = cache_folder() / parent_role / f'{block:04d}.pt'
            meta = read(path.with_suffix('.json'))
            if not path.is_file() or meta['identity'] != read(OWNER / 'plan.json')['required_cache_key']:
                raise ValueError('Existing shared cache is incomplete')
            for index in selected:
                if meta['members'].get(str(index)) != lookup[index]['member']:
                    raise ValueError('Cached membership mismatch')
        result[role] = dict(count=len(ids), chunks=len(groups), tensor_payloads_opened=0)
    return result


def load_rows(role, indices=None, halted=lambda: False, beat=lambda **kw: None):
    parent_role, parent_ids = role_indices(role)
    selected = list(range(len(parent_ids))) if indices is None else list(indices)
    if any(type(i) is not int or i < 0 or i >= len(parent_ids) for i in selected):
        raise IndexError('Subset-local index out of range')
    requested = [parent_ids[i] for i in selected]
    expected = records(role)
    groups, loaded = {}, {}
    for local, index in zip(selected, requested):
        groups.setdefault(index // 256, []).append((local, index))
    for block, entries in groups.items():
        if halted():
            return None
        path = cache_folder() / parent_role / f'{block:04d}.pt'
        meta = read(path.with_suffix('.json'))
        if meta['identity'] != read(OWNER / 'plan.json')['required_cache_key'] or sha(path) != meta['sha256']:
            raise ValueError('Existing CPU cache checksum mismatch')
        chunk = torch.load(path, map_location='cpu', weights_only=False)
        for local, index in entries:
            if index not in chunk:
                raise ValueError('Missing cache member; raw preprocessing is forbidden')
            row = chunk[index]
            if list(row['member']) != expected[local]['member'] or meta['members'].get(str(index)) != expected[local]['member']:
                raise ValueError('Cached parent/subset member mapping changed')
            g = row['graph']
            if not 0 < g['n'] <= 120 or g['features'].shape != (g['n'], 16):
                raise ValueError('Unsupported cached feature dimensions')
            if (g['features'].dtype != torch.float32 or (g['mask'] != 1).any()
                    or (g['constraints'][:, 2] != 0).any()):
                raise ValueError('Cached original-area/no-MIB/all-prediction policy changed')
            loaded[local] = row
        beat(activity='loading existing compatible CPU tensor cache', cache_role=role,
             loaded=len(loaded), count=len(selected), cache_built=0, cache_reused=len(loaded))
    return [loaded[i] for i in selected]


def register_resident(pack):
    for i, row in enumerate(pack.rows):
        _resident[id(row)] = pack, i


def collate(rows, device):
    if str(device).startswith('cuda') and rows and id(rows[0]) in _resident:
        entries = [_resident[id(r)] for r in rows]
        pack = entries[0][0]
        if any(p is not pack for p, _ in entries):
            raise ValueError('Mixed resident roles')
        return pack.batch([i for _, i in entries]), None
    return ResidentRows(rows, device, max_nodes=120).batch(range(len(rows))), None
