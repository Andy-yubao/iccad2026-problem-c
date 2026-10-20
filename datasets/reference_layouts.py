"""Versioned original-reference data; public input transform and sealed tests.

No teacher layouts, model features, losses, or sampling policies live here.
"""
import hashlib
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
import threading

import numpy as np
import torch
from .paths import ROOT, dataset_paths
from .worksets import FrozenMemberReader, ROWS_PER_SHARD
from .cases import _build_case

REGISTRY = Path(os.environ.get('E16_REGISTRY', ROOT / 'protocol/reference_workset_v5.json'))
VERSION = 'E16.reference_balanced.v1'
TASK = 'E14.mean_area.v1'


def digest(value):
    # JSON object keys become strings on disk; normalize before sorting so
    # numeric module-count keys have identical hashes before/after reload.
    value = json.loads(json.dumps(value, ensure_ascii=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_area_inputs(inputs):
    """Same public-only E14.mean_area.v1 rule, without teacher dependencies."""
    values = tuple(v.clone() if hasattr(v, 'clone') else v for v in inputs)
    _, areas, _, _, _, constraints, specs = values
    c, s = np.asarray(constraints), np.asarray(specs)
    locked = c[:, :2].any(1)
    for group in sorted(set(c[:, 2]) - {0}):
        ids = np.flatnonzero(c[:, 2] == group)
        areas[ids] = float(areas[ids].double().mean())
        rigid = ids[locked[ids]]
        if len(rigid):
            wh = s[rigid[0], 2:]
            if not np.allclose(s[rigid, 2:], wh, rtol=0, atol=1e-10):
                raise ValueError('locked_dimensions_disagree')
            if abs(float(wh.prod()) / float(areas[ids[0]]) - 1) >= .009999:
                raise ValueError('mean_area_incompatible_with_locked_shape')
    return values


def public_hard_reference(areas, constraints, specs):
    """Reference substitute sufficient for existing hard checks, no free labels."""
    areas, c, s = np.asarray(areas), np.asarray(constraints), np.asarray(specs)
    reference = np.column_stack((np.zeros((len(areas), 2)), np.sqrt(areas), np.sqrt(areas)))
    locked, pre = c[:, :2].any(1), c[:, 1] != 0
    reference[locked, 2:] = s[locked, 2:]
    reference[pre] = s[pre]
    return reference.tolist()


def inspect_shard(shard, parent, *, sample_rows=None):
    """Public-only candidate metadata; never access free positions or metrics.

    Original files bundle labels with inputs, so torch deserializes their
    container. Selection indexes only areas, constraints, and public locked WH.
    sample_rows exists solely for a <=5-case implementation check.
    """
    split = json.loads((ROOT / 'protocol/split_v1.json').read_text(encoding='utf-8'))
    if shard not in split['shards'][parent]:
        raise ValueError('Candidate outside original partition')
    root = dataset_paths()['training_root']
    path = (root / shard).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Shard escapes data root')
    stat = path.stat()
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if len(raw) < 7 or len(raw[0]) != ROWS_PER_SHARD:
        raise ValueError('Unexpected raw schema')
    accepted, rejected, counts = [], Counter(), Counter()
    indices = range(ROWS_PER_SHARD) if sample_rows is None else sample_rows
    for row in indices:
        f = raw[0][row].detach().cpu().float()
        if f.ndim != 2 or f.shape[1] < 6 or not torch.isfinite(f).all() or (f[:, 0] <= 0).any():
            raise ValueError('Invalid public block features')
        n, areas, constraints = len(f), f[:, 0], f[:, 1:]
        counts[n] += 1
        locked = constraints[:, :2].bool().any(1)
        specs = torch.full((n, 4), -1., dtype=torch.float32)
        # In the raw WHXY schema only locked widths/heights are public here.
        specs[locked, 2:] = raw[5][row][locked, :2].float()
        if not torch.isfinite(specs).all() or (specs[locked, 2:] <= 0).any():
            raise ValueError('Invalid public locked shape')
        try:
            mean_area_inputs((n, areas, None, None, None, constraints, specs))
        except ValueError as error:
            if str(error) not in ('locked_dimensions_disagree', 'mean_area_incompatible_with_locked_shape'):
                raise
            rejected[str(error)] += 1
            continue
        accepted.append([int(row), n])
    return {'shard': shard, 'parent': parent, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'rows_examined': sum(counts.values()), 'raw_n_counts': dict(counts),
            'accepted': accepted, 'rejections': dict(rejected),
            'selection_uses_free_labels': False, 'selection_uses_Q': False}


@lru_cache(maxsize=4)
def _registry(path_string, stat_size, stat_mtime):
    data = json.loads(Path(path_string).read_text(encoding='utf-8'))
    expected = dict(data); checksum = expected.pop('manifest_sha256')
    if digest(expected) != checksum or data['dataset_version'] != VERSION or data['task_version'] != TASK:
        raise ValueError('Registry checksum/version mismatch')
    for name, value in data['frozen_original_hashes'].items():
        if sha(ROOT / name) != value:
            raise ValueError('Frozen original changed: ' + name)
    split = json.loads((ROOT / 'protocol/split_v1.json').read_text(encoding='utf-8'))
    main = {}
    for role in ('train', 'dev', 'local_test'):
        records = data['sets'][role]['records']
        members = [tuple(r['member']) for r in records]
        if len(members) != len(set(members)) or len(members) != data['sets'][role]['count']:
            raise ValueError('Duplicate or inconsistent members')
        allowed = set(split['shards'][role])
        if any(s not in allowed or not 0 <= r < ROWS_PER_SHARD for s, r in members):
            raise ValueError('Member violates original shard boundary')
        main[role] = set(members)
    if any(main[a] & main[b] for a, b in (('train', 'dev'), ('train', 'local_test'), ('dev', 'local_test'))):
        raise ValueError('Partition overlap')
    for role, parent in (('train_monitor500', 'train'), ('dev_monitor500', 'dev')):
        members = [tuple(r['member']) for r in data['sets'][role]['records']]
        if len(set(members)) != len(members) or not set(members) <= main[parent]:
            raise ValueError('Invalid monitor subset')
    if not data.get('pilot'):
        if [data['sets'][r]['count'] for r in ('train', 'dev', 'local_test', 'train_monitor500', 'dev_monitor500')] != [16000, 2000, 2000, 500, 500]:
            raise ValueError('Wrong formal set sizes')
        hist = {r: Counter(x['n'] for x in data['sets'][r]['records']) for r in main}
        if hist['dev'] != hist['local_test'] or hist['train'] != Counter({n: 8*c for n, c in hist['dev'].items()}):
            raise ValueError('Module-count distributions are not exactly 8:1:1')
    root = dataset_paths()['training_root']
    for shard, entry in data.get('source_shards', {}).items():
        path = (root/shard).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source shard escapes data root')
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('Frozen source shard changed: '+shard)
    return data


def registry(path=REGISTRY):
    path = Path(path).resolve(); stat = path.stat()
    return _registry(str(path), stat.st_size, stat.st_mtime_ns)


class ReferenceLayouts:
    def __init__(self, role, *, registry_path=REGISTRY):
        if role not in ('train', 'dev', 'train_monitor500', 'dev_monitor500'):
            raise PermissionError('Test labels and model evaluation remain sealed')
        self.registry = registry(registry_path)
        self.role = role
        self.records = self.registry['sets'][role]['records']
        parent = 'train' if role.startswith('train') else 'dev'
        self.parent = parent
        self._local = threading.local()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if not hasattr(self._local, 'reader'):
            self._local.reader = FrozenMemberReader([r['member'] for r in self.records], self.parent,
                                                    identity=self.registry['manifest_sha256'])
        raw = self._local.reader.case(index)
        record = self.records[index]
        if raw.block_count != record['n']:
            raise ValueError('Block count differs from frozen metadata')
        inputs = mean_area_inputs(raw.solver_inputs)
        case = _build_case(raw.case_id, raw.source, raw.solver_inputs[1:6],
                           raw.reference_positions, raw.stored_metrics)
        case.solver_inputs = inputs
        # Original supervision, never a solver/teacher output. Its WH need not
        # match transformed areas, so only center/log-ratio anchors are trained.
        return {'member': tuple(record['member']), 'case': case,
                'positions': np.asarray(raw.reference_positions, dtype=np.float32)}

    def archive_item(self, index):
        """Cache-builder interface; this reads original data, not an archive."""
        return self[index]
