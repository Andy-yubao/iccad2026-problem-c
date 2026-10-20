"""One shared, resumable columnar cache; memory-map instead of 16k unpickles."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from datasets.quality_layouts import QualityLayouts, REGISTRY, public_hard_reference
from datasets.cases import ResearchCase
from .data import prepare
from .io import ROOT, HERE, sha, save, replace, exclusive

FORMAT = 'E15.packed_public_v1'
DEFAULT = HERE/'runs/prepared_v4_cache_v1'
NODES = ('features', 'mask', 'known', 'area', 'specs', 'constraints')
PAIRS = ('adjacency', 'mib', 'group')


def identity():
    names = ['datasets/quality_layouts.py', 'experiments/E15_quality_curriculum/features.py',
             'experiments/E15_quality_curriculum/data.py', 'experiments/E15_quality_curriculum/cache.py']
    return {'format': FORMAT, 'registry': sha(REGISTRY), 'paths': sha(ROOT/'protocol/paths.json'),
            'preparation_sources': {name: sha(ROOT/name) for name in names}}


def pack(rows):
    columns = {}
    for name in NODES:
        columns['g_'+name] = np.concatenate([r['graph'][name].numpy() for r in rows])
    for name in PAIRS:
        columns['g_'+name] = np.concatenate([r['graph'][name].numpy().reshape(-1) for r in rows])
    columns['geometry'] = np.concatenate([r['target']['geometry'].numpy() for r in rows])
    columns['scale'] = np.array([float(r['graph']['scale']) for r in rows], dtype=np.float32)
    columns['anchors'] = np.array([[r['case'].baseline_metrics['hpwl_baseline'],
                                   r['case'].baseline_metrics['area_baseline']] for r in rows], dtype=np.float64)
    for key, at in (('raw_areas', 1), ('raw_specs', 6), ('raw_pins', 4)):
        columns[key] = np.concatenate([np.asarray(r['case'].solver_inputs[at]) for r in rows])
    for name in ('b2b', 'p2b'):
        columns[name] = np.concatenate([r['target'][name].numpy() for r in rows])
    columns['counts'] = np.array([[len(r['graph']['valid']), len(r['target']['b2b']),
        len(r['target']['p2b']), len(r['case'].solver_inputs[4])] for r in rows], dtype=np.int64)
    return columns


def build_role(role, output, *, indices=None, workers=4, chunk_size=128):
    dataset = QualityLayouts(role)
    ids = list(range(len(dataset))) if indices is None else list(indices)
    expected = {'identity': identity(), 'role': role, 'indices': ids,
                'members': [dataset.records[i]['member'] for i in ids], 'pilot': indices is not None}
    output = Path(output)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['expected'] != expected:
            raise ValueError('Cache identity differs; use a new cache directory')
        verify(output, manifest)
        print(f'Cache ready: {role}, {len(ids)} cases', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    preparing = output/'preparing.json'
    if preparing.exists() and json.loads(preparing.read_text(encoding='utf-8')) != expected:
        raise ValueError('Interrupted cache has a different identity')
    save(preparing, expected)
    parts = output/'parts'
    parts.mkdir(exist_ok=True)
    begin = time.perf_counter()
    def read(index):
        return prepare(dataset.archive_item(index))
    part_paths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(ids), chunk_size):
            path = parts/f'{start:06d}.npz'
            if not path.exists():
                rows = list(pool.map(read, ids[start:start+chunk_size]))
                columns = pack(rows)
                temp = path.with_suffix('.tmp')
                with temp.open('wb') as handle:
                    np.savez(handle, **columns)
                replace(temp, path)
            part_paths.append(path)
            count = min(start+chunk_size, len(ids))
            seconds = time.perf_counter()-begin
            print(f'Cache {role}: {count}/{len(ids)}, elapsed={seconds:.1f}s', flush=True)
    # Concatenate compact columns once. Open each chunk once per merge pass;
    # no per-layout tensor serialization remains in the completed cache.
    descriptions = {}
    for path in part_paths:
        with np.load(path, allow_pickle=False) as part:
            for key in part.files:
                value = part[key]
                if key not in descriptions:
                    descriptions[key] = [0, value.shape[1:], value.dtype]
                descriptions[key][0] += len(value)
    arrays = {}
    for key, (count, shape, dtype) in descriptions.items():
        arrays[key] = np.lib.format.open_memmap(output/(key+'.npy.tmp'), mode='w+',
                                               dtype=dtype, shape=(count, *shape))
    cursors = dict.fromkeys(arrays, 0)
    for path in part_paths:
        with np.load(path, allow_pickle=False) as part:
            for key, target in arrays.items():
                value = part[key]
                start = cursors[key]
                target[start:start+len(value)] = value
                cursors[key] += len(value)
    for key in list(arrays):
        arrays[key].flush()
        del arrays[key]
    del target
    files = {}
    for key in descriptions:
        path = output/(key+'.npy')
        replace(output/(key+'.npy.tmp'), path)
        stat = path.stat()
        files[path.name] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha(path)}
    save(manifest_path, {'expected': expected, 'files': files, 'count': len(ids),
                        'build_seconds_this_invocation': time.perf_counter()-begin})
    # These are reproducible staging chunks. Final mmap columns plus their
    # checksums are complete before removing the redundant local parts.
    for path in part_paths:
        path.unlink()
    parts.rmdir()
    print(f'Cache completed: {role}, {len(ids)} cases', flush=True)


def verify(output, manifest):
    for name, entry in manifest['files'].items():
        stat = (output/name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('Cache changed: '+name)


class PackedRows:
    def __init__(self, role, indices=None, *, cache_root=DEFAULT, allow_pilot=False):
        started = time.perf_counter()
        dataset = QualityLayouts(role)
        base = 'train' if role == 'train_monitor500' else 'dev' if role == 'dev_monitor500' else role
        if base not in ('train', 'dev'):
            raise PermissionError('Training cache never exposes sealed tests')
        path = Path(cache_root)/base
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        if manifest['expected']['identity'] != identity() or (manifest['expected']['pilot'] and not allow_pilot):
            raise ValueError('Cache identity/purpose mismatch; run cache --prepare')
        verify(path, manifest)
        members = {tuple(m): i for i, m in enumerate(manifest['expected']['members'])}
        selected = range(len(dataset)) if indices is None else indices
        self.records = [dataset.records[i] for i in selected]
        self.ids = [members[tuple(record['member'])] for record in self.records]
        self.arrays = {name[:-4]: np.load(path/name, mmap_mode='c', allow_pickle=False) for name in manifest['files']}
        counts = self.arrays['counts']
        self.offsets = [np.r_[0, np.cumsum(counts[:, col])] for col in range(4)]
        self.pair_offsets = np.r_[0, np.cumsum(counts[:, 0]**2)]
        print(f'Memory-mapped {role}: {len(self.ids)} cases in {time.perf_counter()-started:.3f}s', flush=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        slices = [slice(offset[i], offset[i+1]) for offset in self.offsets]
        a = self.arrays
        n = int(a['counts'][i, 0])
        graph = {name: torch.from_numpy(a['g_'+name][slices[0]]) for name in NODES}
        for name in PAIRS:
            graph[name] = torch.from_numpy(a['g_'+name][self.pair_offsets[i]:self.pair_offsets[i+1]].reshape(n, n))
        graph['valid'] = torch.ones(n, dtype=torch.bool)
        graph['scale'] = torch.tensor(float(a['scale'][i]), dtype=torch.float32)
        anchors = a['anchors'][i]
        scale = float(graph['scale'])
        c = graph['constraints']
        target = {'geometry': torch.from_numpy(a['geometry'][slices[0]]),
                  'wire_anchor': float(anchors[0])/scale, 'area_anchor': float(anchors[1])/float(graph['scale'].square()),
                  'b2b': torch.from_numpy(a['b2b'][slices[1]]), 'p2b': torch.from_numpy(a['p2b'][slices[2]]),
                  'pins': torch.from_numpy(a['raw_pins'][slices[3]])/graph['scale'],
                  'groups': [torch.where(c[:, 3] == value)[0] for value in c[:, 3].unique().tolist() if value > 0],
                  'mibs': [torch.where(c[:, 2] == value)[0] for value in c[:, 2].unique().tolist() if value > 0]}
        record = self.records[index]
        inputs = (n, torch.from_numpy(a['raw_areas'][slices[0]]), target['b2b'], target['p2b'],
                  torch.from_numpy(a['raw_pins'][slices[3]]), c.float(), torch.from_numpy(a['raw_specs'][slices[0]]))
        case = ResearchCase(record['case_id'], {'member': record['member'], 'reference_scope': 'public hard specs only'},
            inputs, public_hard_reference(inputs[1], c, inputs[6]),
            {'hpwl_baseline': float(anchors[0]), 'area_baseline': float(anchors[1])})
        return {'graph': graph, 'target': target, 'member': tuple(record['member']), 'case': case}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Use 1..8 cache preparation workers')
    torch.set_num_threads(2)
    with exclusive(DEFAULT/'prepare.lock'):
        for role in ('train', 'dev'):
            build_role(role, DEFAULT/role, workers=args.workers)


if __name__ == '__main__':
    main()
