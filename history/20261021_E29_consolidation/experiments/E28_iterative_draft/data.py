"""Read-only frozen E17 tensor reuse plus E28 public terminal/draft supplements."""
import hashlib
import json
import operator
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts, REGISTRY
from .io import ROOT, HERE, sha
from .batches import collate as base_collate

SOURCE_CACHE = ROOT / 'experiments/E25_integrated_baseline/runs/prepared_v5_e10_cache_v1'
CACHE = HERE / 'runs/prepared_v5_e28_v1'
BASE_CHECKPOINT = ROOT / 'experiments/E25_integrated_baseline/runs/20261020_e10_geometry_v5_seed0/checkpoints/best.pt'
_loaded = {}


def identity():
    provenance = json.loads((HERE / 'provenance.json').read_text(encoding='utf-8'))
    if sha(BASE_CHECKPOINT) != provenance['base_checkpoint_sha256']:
        raise ValueError('Frozen base checkpoint changed')
    if sha(REGISTRY) != provenance['registry_sha256']:
        raise ValueError('Frozen v5 membership changed')
    current = {'revision': 'E28.public_terminal_draft.v1', 'registry': sha(REGISTRY),
            'paths': sha(ROOT / 'protocol/paths.json'), 'base': provenance['base_checkpoint_sha256'],
            'sources': {name: sha(HERE / name) for name in
                        ('base_model.py', 'features.py', 'batches.py', 'model.py', 'data.py', 'prepare.py')},
            'datasets': {p.name: sha(p) for p in sorted((ROOT / 'datasets').glob('*.py'))}}
    compatibility = HERE / 'cache_compatibility.json'
    if compatibility.exists():
        return compatible_identity(current, json.loads(compatibility.read_text(encoding='utf-8')))
    return current


def compatible_identity(current, record):
    """Reuse only the exactly reviewed metadata-order repair, never arbitrary code."""
    approved = record['approved_repaired_sources']
    if set(approved) != {'data.py', 'prepare.py'}:
        raise ValueError('Invalid cache compatibility scope')
    adjusted = {**current, 'sources': dict(current['sources'])}
    legacy = record['artifact_identity']
    for name, digest in approved.items():
        if current['sources'][name] != digest:
            raise ValueError('Cache compatibility approval does not cover current source: ' + name)
        adjusted['sources'][name] = legacy['sources'][name]
    if adjusted != legacy:
        raise ValueError('Model, numeric inputs, dataset or unapproved cache source changed')
    # Keep the immutable creation identity in already-written tensor chunks.
    # Actual repaired generation/reader sources are recorded separately.
    return legacy


def part_indices(original, members):
    """The source cache is shard-ordered; indices map rows to frozen workset order."""
    indices = [operator.index(index) for index in original['indices']]
    if len(indices) != len(original['rows']) or len(indices) != len(set(indices)):
        raise ValueError('Source part indices are missing or duplicated')
    if any(index < 0 or index >= len(members) for index in indices):
        raise ValueError('Source part index outside frozen workset')
    expected = [list(members[index]) for index in indices]
    if [list(row['member']) for row in original['rows']] != expected:
        raise ValueError('Source rows do not match their frozen indices')
    return indices


def place_rows(ordered, indices, rows):
    if len(indices) != len(rows):
        raise ValueError('Part index/row count differs')
    for index, row in zip(indices, rows):
        if not 0 <= index < len(ordered) or ordered[index] is not None:
            raise ValueError('Duplicate or out-of-range cache index across parts')
        ordered[index] = row


def validate_supplement(saved, original, entry, members, ident):
    if saved['identity'] != ident or saved['source_part'] != entry:
        raise ValueError('Incomplete/incompatible prepared part')
    indices = part_indices(original, members)
    if 'indices' in saved and saved['indices'] != indices:
        raise ValueError('Supplement index mapping differs from original cache')
    if [row['member'] for row in saved['rows']] != [list(members[index]) for index in indices]:
        raise ValueError('Supplement members differ from original index mapping')
    return indices


def source_manifest(role):
    if role not in ('train', 'dev'):
        raise PermissionError('Test data remain sealed')
    expected = json.loads((HERE / 'provenance.json').read_text(encoding='utf-8'))['source_cache_identity']
    if expected['registry'] != sha(REGISTRY) or expected['paths'] != sha(ROOT / 'protocol/paths.json'):
        raise ValueError('Original cache data identity changed')
    if any(sha(ROOT / 'datasets' / name) != value for name, value in expected['datasets_sources'].items()):
        raise ValueError('Original cache reader changed')
    folder = SOURCE_CACHE / role
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != expected:
        raise ValueError('Wrong original cache')
    if manifest['members'] != [r['member'] for r in ReferenceLayouts(role).records]:
        raise ValueError('Original cache member order changed')
    for entry in manifest['parts']:
        stat = (folder / entry['name']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('Original cached part changed')
    return manifest


def augment(row):
    """Only public inputs determine terminals, random keys and the draft."""
    graph = dict(row['graph'])
    inputs = row['case'].solver_inputs
    n, _, _, p2b, pins, constraints, _ = inputs
    graph['pin_xy'] = torch.as_tensor(pins, dtype=torch.float32)[:, :2] / graph['scale']
    graph['pin_weights'] = torch.zeros(n, len(pins))
    graph['block_weights'] = torch.zeros(n, n)
    block_edges = torch.as_tensor(inputs[2])
    block_edges = block_edges[~(block_edges == -1).all(1)]
    if block_edges.numel():
        i, j = block_edges[:, 0].long(), block_edges[:, 1].long()
        graph['block_weights'].index_put_((i, j), block_edges[:, 2].float(), accumulate=True)
        graph['block_weights'].index_put_((j, i), block_edges[:, 2].float(), accumulate=True)
    edges = torch.as_tensor(p2b)
    edges = edges[~(edges == -1).all(1)]
    if edges.numel():
        graph['pin_weights'].index_put_((edges[:, 1].long(), edges[:, 0].long()), edges[:, 2].float(), accumulate=True)
    # At least one masked dummy terminal permits graphs with no external terminals.
    if not len(pins):
        graph['pin_xy'] = torch.zeros(1, 2)
        graph['pin_weights'] = torch.zeros(n, 1)
    neighbors = (graph['pin_weights'] > 0).sum(-1)
    max_links = max(1, int(neighbors.max()))
    graph['pin_indices'] = torch.zeros(n, max_links, dtype=torch.long)
    graph['pin_link_weights'] = torch.zeros(n, max_links)
    for index in range(n):
        linked = (graph['pin_weights'][index] > 0).nonzero().flatten()
        graph['pin_indices'][index, :len(linked)] = linked
        graph['pin_link_weights'][index, :len(linked)] = graph['pin_weights'][index, linked]
    # Random keys depend on public numerical inputs, never dataset/file identity.
    public_digest = hashlib.sha256(b'E28.latent.v1')
    for value in inputs:
        array = np.asarray(value, dtype='<f8')
        public_digest.update(str(array.shape).encode())
        public_digest.update(array.tobytes())
    seed = int(public_digest.hexdigest()[:15], 16)
    graph['latent'] = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    # Original ratio anchors are used only where original WH matches current area.
    original_area = graph['reference'][:, 2:].prod(-1)
    graph['shape_anchor_mask'] = ((original_area / graph['area'].clamp_min(1e-12) - 1).abs() <= .01).float()
    graph['group_ids'] = torch.as_tensor(constraints)[:, 3].long()
    return {**row, 'graph': graph}


def contact_tree(row, draft):
    """Initial-draft spanning trees are fixed public topology proxies, never labels."""
    from .base_model import decode
    from .batches import PUBLIC_KEYS
    g = {k: row['graph'][k][None] for k in PUBLIC_KEYS if k != 'valid'}
    g['valid'] = torch.ones(1, len(draft), dtype=torch.bool)
    boxes = decode(draft[None], g)[0]
    hi = boxes[:, :2] + boxes[:, 2:]
    gap = (torch.maximum(boxes[:, None, :2], boxes[None, :, :2]) -
           torch.minimum(hi[:, None], hi[None, :])).clamp_min(0).square().sum(-1)
    tree = torch.zeros(len(draft), len(draft), dtype=torch.bool)
    ids = row['graph']['group_ids']
    for group in ids.unique().tolist():
        if group <= 0:
            continue
        members = (ids == group).nonzero().flatten().tolist()
        visited, remaining = [members[0]], set(members[1:])
        while remaining:
            _, i, j = min((float(gap[i, j]), i, j) for i in visited for j in remaining)
            tree[i, j] = True
            visited.append(j)
            remaining.remove(j)
    return tree


def public_row(inputs, member):
    """Build an inference row from allowed inputs only; labels are placeholders."""
    from .features import graph_features
    f = graph_features(inputs); scale = f['scale']; n = int(inputs[0])
    known = torch.cat(((f['specs'][:, :2] + f['specs'][:, 2:] / 2) / scale,
                       (f['specs'][:, 2].clamp_min(1e-8) / f['specs'][:, 3].clamp_min(1e-8)).log()[:, None]), -1)
    graph = {'features': f['features'], 'adjacency': f['adjacency'], 'mib': f['relations'][0],
             'group': f['relations'][1], 'mask': f['mask'], 'known': known * (1 - f['mask']),
             'area': torch.where(f['locked'], f['specs'][:, 2:].prod(1), f['areas']) / scale.square(),
             'specs': torch.where(f['locked'][:, None], f['specs'] / scale, torch.zeros_like(f['specs'])),
             'constraints': torch.as_tensor(inputs[5]).long(), 'scale': scale, 'n': n,
             'target': torch.zeros(n, 3), 'reference': torch.zeros(n, 4)}
    class PublicCase:
        solver_inputs = inputs
    return augment({'graph': graph, 'member': member, 'case': PublicCase()})


def collate(rows, device, *, online=False):
    graph = base_collate([r['graph'] for r in rows], range(len(rows)), 'cpu')
    b, n = graph['valid'].shape
    m = max(len(r['graph']['pin_xy']) for r in rows)
    links = max(r['graph']['pin_indices'].shape[1] for r in rows)
    graph.update(pin_xy=torch.zeros(b, m, 2), pin_weights=torch.zeros(b, n, m),
                 pin_indices=torch.zeros(b, n, links, dtype=torch.long), pin_link_weights=torch.zeros(b, n, links),
                 pin_valid=torch.zeros(b, m, dtype=torch.bool), latent=torch.zeros(b, n, 2),
                 shape_anchor_mask=torch.zeros(b, n), contact_tree=torch.zeros(b, n, n, dtype=torch.bool),
                 block_weights=torch.zeros(b, n, n))
    cached = all('draft' in r['graph'] for r in rows) and not online
    if cached:
        graph['draft'] = torch.zeros(b, n, 3)
    for index, row in enumerate(rows):
        source = row['graph']; ni, mi = source['n'], len(source['pin_xy'])
        graph['pin_xy'][index, :mi] = source['pin_xy']
        graph['pin_weights'][index, :ni, :mi] = source['pin_weights']
        ki = source['pin_indices'].shape[1]
        graph['pin_indices'][index, :ni, :ki] = source['pin_indices']
        graph['pin_link_weights'][index, :ni, :ki] = source['pin_link_weights']
        graph['pin_valid'][index, :mi] = True
        graph['latent'][index, :ni] = source['latent']
        graph['shape_anchor_mask'][index, :ni] = source['shape_anchor_mask']
        graph['block_weights'][index, :ni, :ni] = source['block_weights']
        if 'contact_tree' in source:
            graph['contact_tree'][index, :ni, :ni] = source['contact_tree']
        if cached:
            graph['draft'][index, :ni] = source['draft']
    return {k: value.to(device) for k, value in graph.items()}


def load_rows(role):
    if role not in ('train', 'dev', 'train_monitor500', 'dev_monitor500'):
        raise PermissionError('Test roles are sealed')
    parent = 'train' if role.startswith('train') else 'dev'
    if parent not in _loaded:
        manifest = json.loads((CACHE / parent / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['identity'] != identity():
            raise ValueError('E28 supplement identity mismatch; prepare a new version')
        if manifest['members'] != [r['member'] for r in ReferenceLayouts(parent).records]:
            raise ValueError('E28 cache member order mismatch')
        source = source_manifest(parent)
        if manifest['source_manifest_sha256'] != sha(SOURCE_CACHE / parent / 'manifest.json'):
            raise ValueError('Original cache manifest changed')
        rows = [None] * len(manifest['members'])
        for entry in manifest['parts']:
            path = CACHE / parent / entry['name']; stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
                raise ValueError('E28 cache part changed')
            value = torch.load(path, map_location='cpu', weights_only=False)
            original = torch.load(SOURCE_CACHE / parent / entry['name'], map_location='cpu', weights_only=False)
            if original['identity'] != source['identity'] or len(original['rows']) != len(value['rows']):
                raise ValueError('Original part identity or length differs')
            source_entry = next(item for item in source['parts'] if item['name'] == entry['name'])
            indices = validate_supplement(value, original, source_entry, manifest['members'], manifest['identity'])
            combined = []
            for row, supplement in zip(original['rows'], value['rows']):
                if list(row['member']) != supplement['member']:
                    raise ValueError('Supplement attached to another member')
                combined.append({**row, 'graph': {**row['graph'], **supplement['graph']}})
            place_rows(rows, indices, combined)
        if any(row is None for row in rows) or [list(r['member']) for r in rows] != manifest['members']:
            raise ValueError('E28 cached payload members differ')
        _loaded[parent] = {tuple(r['member']): r for r in rows}
    return [_loaded[parent][tuple(r['member'])] for r in ReferenceLayouts(role).records]
