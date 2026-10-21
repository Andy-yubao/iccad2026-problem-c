"""Resumable public-only supplements; original tensors are reused read-only."""
import argparse
import json
import os
import time
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '4')
import torch
from datasets.reference_layouts import ReferenceLayouts
from .io import HERE, ROOT, sha, save, checkpoint, exclusive, snapshot, sources
from .data import (SOURCE_CACHE, CACHE, BASE_CHECKPOINT, source_manifest, identity, augment,
                   collate, contact_tree, part_indices, place_rows, validate_supplement)
from .model import DraftModel, inputs_only

SUPPLEMENT_KEYS = ('pin_xy', 'pin_weights', 'pin_indices', 'pin_link_weights', 'block_weights',
                   'latent', 'shape_anchor_mask', 'draft', 'contact_tree')


def prepare(device='cuda'):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if os.environ.get('E16_REGISTRY'):
        raise ValueError('Remove registry override for formal preparation')
    begin = time.perf_counter()
    with exclusive(CACHE.with_suffix('.lock')):
        ident = identity()
        if (CACHE / 'identity.json').exists():
            if json.loads((CACHE / 'identity.json').read_text(encoding='utf-8')) != ident:
                raise ValueError('Preparation identity changed; do not overwrite cache')
        else:
            save(CACHE / 'identity.json', ident)
        if (CACHE / 'results.json').exists():
            for role in ('train', 'dev'):
                source_manifest(role)
                manifest = json.loads((CACHE / role / 'manifest.json').read_text(encoding='utf-8'))
                if manifest['identity'] != ident or manifest['source_manifest_sha256'] != sha(SOURCE_CACHE / role / 'manifest.json'):
                    raise ValueError('Completed supplement identity changed')
                for entry in manifest['parts']:
                    stat = (CACHE / role / entry['name']).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
                        raise ValueError('Completed supplement part changed')
            print('Reusing complete registered E28 supplements', flush=True)
            return True
        model = DraftModel().to(device).eval()
        base = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=False)
        model.base.load_state_dict(base['model'])
        latest = torch.load(BASE_CHECKPOINT.parent / 'latest.pt', map_location='cpu', weights_only=False)
        if not latest['finished'] or latest['identity']['implementation_check']:
            raise ValueError('Base training is not formally frozen')
        for role in ('train', 'dev'):
            manifest = source_manifest(role)
            members = [r['member'] for r in ReferenceLayouts(role).records]
            folder = CACHE / role; folder.mkdir(parents=True, exist_ok=True)
            parts = []
            observed = [None] * len(members)
            for entry in manifest['parts']:
                if (CACHE / 'STOP').exists():
                    save(CACHE / 'progress.json', {'status': 'user_stop', 'role': role,
                         'completed_parts': len(parts), 'seconds_this_invocation': time.perf_counter() - begin,
                         'test_read': False})
                    return False
                path = folder / entry['name']
                original = torch.load(SOURCE_CACHE / role / entry['name'], map_location='cpu', weights_only=False)
                if original['identity'] != manifest['identity']:
                    raise ValueError('Original part identity differs')
                indices = part_indices(original, members)
                reused = path.exists()
                if reused:
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    validate_supplement(saved, original, entry, members, ident)
                else:
                    rows = [augment(row) for row in original['rows']]
                    with torch.no_grad():
                        for start in range(0, len(rows), 32):
                            batch = rows[start:start + 32]
                            graph = collate(batch, device)
                            drafts = model.initial(inputs_only(graph)).cpu()
                            for row, draft in zip(batch, drafts):
                                row['graph']['draft'] = draft[:row['graph']['n']].clone()
                                row['graph']['contact_tree'] = contact_tree(row, row['graph']['draft'])
                    saved = {'identity': ident, 'source_part': entry, 'indices': indices,
                             'generation_sources': {name: sha(HERE / name) for name in ('data.py', 'prepare.py')},
                             'rows': [{'member': list(row['member']),
                                       'graph': {key: row['graph'][key] for key in SUPPLEMENT_KEYS}} for row in rows]}
                    checkpoint(path, saved)
                place_rows(observed, indices, [row['member'] for row in saved['rows']])
                stat = path.stat()
                parts.append({'name': entry['name'], 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
                save(CACHE / 'progress.json', {'status': 'running', 'role': role,
                     'completed_parts': len(parts), 'total_parts': len(manifest['parts']),
                     'reused_current_part': reused, 'device': device, 'cpu_threads': 4, 'test_read': False})
                action = 'reused' if reused else 'computed'
                print(f'prepare {role}: {len(parts)}/{len(manifest["parts"])} ({action})', flush=True)
            # Verify logical workset order via original indices, not disk shard order.
            if observed != members:
                raise ValueError('Supplement is not the complete frozen partition')
            save(folder / 'manifest.json', {'identity': ident, 'members': members, 'parts': parts,
                 'reader_generation_sources': {name: sha(HERE / name) for name in ('data.py', 'prepare.py')},
                 'source_manifest_sha256': sha(SOURCE_CACHE / role / 'manifest.json')})
        save(CACHE / 'results.json', {'status': 'complete', 'device': device, 'cpu_threads': 4,
             'seconds_this_invocation': time.perf_counter() - begin, 'test_read': False,
             'source_cache': str(SOURCE_CACHE), 'supplements_only': True})
        snapshot(CACHE / 'source_snapshot.zip', {'sources': sources()})
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    prepare(parser.parse_args().device)
