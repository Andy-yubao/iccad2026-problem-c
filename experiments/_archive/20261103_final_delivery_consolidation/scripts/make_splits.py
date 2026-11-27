"""Build/check the fixed 80/10/10 shard split without loading tensor payloads.

Default is read-only verification. --create writes once and refuses replacement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.paths import dataset_paths

TARGET = ROOT / 'protocol/split_v1.json'
SEED = 20261015


def build():
    data_root = dataset_paths(ROOT)['training_root']
    roles = {name: [] for name in ['train', 'dev', 'local_test']}
    test_panel = []
    for idx in range(100):
        worker = f'worker_{idx}'
        files = sorted((data_root / worker).glob('layouts_*.th'))
        if len(files) != 90:
            raise ValueError(f'{worker}: expected 90 shards, found {len(files)}')
        reserved = data_root / worker / 'layouts_0.th'
        if reserved not in files:
            raise ValueError(f'Schema-probe shard missing: {reserved}')
        ranked = sorted([p for p in files if p != reserved],
                        key=lambda p: hashlib.sha256(f'{SEED}|{worker}|{p.name}'.encode()).digest())
        assignment = {'train': ranked[:72], 'dev': [reserved, *ranked[72:80]], 'local_test': ranked[80:]}
        for role, paths in assignment.items():
            roles[role].extend(p.relative_to(data_root).as_posix() for p in paths)
        test_panel.append({'shard': assignment['local_test'][0].relative_to(data_root).as_posix(), 'sample_index': 0})
    dev_smoke = [{'shard': f'worker_{i}/layouts_0.th', 'sample_index': 0} for i in [0, 19, 39, 69, 99]]
    split_hash = hashlib.sha256(json.dumps(roles, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {
        'version': 1, 'seed': SEED,
        # Preserve the historical location field; only resolve the physical root externally.
        'data_root_relative_to_workspace': (json.loads(TARGET.read_text(encoding='utf-8'))['data_root_relative_to_workspace']
                                            if TARGET.exists() else data_root.relative_to(ROOT).as_posix()),
        'method': 'Within each worker, SHA256-ranked whole shards: 72 train, 9 dev, 9 local_test. layouts_0.th is always dev.',
        'grouping_unit': 'entire layouts_*.th shard; never split the 112 layouts within a shard',
        'schema_probe_shards_reserved_for_dev': 'worker_*/layouts_0.th',
        'shards': roles, 'shard_counts': {k: len(v) for k, v in roles.items()},
        'estimated_layout_counts_from_loader_formula': {k: len(v) * 112 for k, v in roles.items()},
        'split_content_sha256': split_hash,
        'dev_smoke_panel': dev_smoke,
        'local_test_panel_100': test_panel,
        'public_compatibility_case_ids': [0, 19, 39, 69, 99],
        'local_test_payloads_opened_at_split_creation': False,
        'limitations': 'Worker IDs are not block counts. Size balance and cross-shard family/duplicate independence are not verified by this metadata-only split.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    current = build()
    sets = {k: set(v) for k, v in current['shards'].items()}
    assert len(set.union(*sets.values())) == 9000
    assert not (sets['train'] & sets['dev'] or sets['train'] & sets['local_test'] or sets['dev'] & sets['local_test'])
    assert all(x['shard'] in sets['dev'] for x in current['dev_smoke_panel'])
    assert all(x['shard'] in sets['local_test'] for x in current['local_test_panel_100'])
    if args.create:
        if TARGET.exists():
            raise FileExistsError('Split already frozen; do not overwrite. A change requires a new protocol version.')
        current['created_at_utc'] = datetime.now(timezone.utc).isoformat()
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8')
    else:
        frozen = json.loads(TARGET.read_text(encoding='utf-8'))
        for key, value in current.items():
            if frozen.get(key) != value:
                raise ValueError(f'Frozen split mismatch in {key}')
    print(json.dumps({'split': str(TARGET), 'counts': current['shard_counts'],
                      'estimated_layouts': current['estimated_layout_counts_from_loader_formula'],
                      'sha256': current['split_content_sha256'], 'tensor_payloads_loaded': 0,
                      'status': 'created' if args.create else 'verified'}, indent=2))


if __name__ == '__main__':
    main()
