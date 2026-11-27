"""Fresh module-count-balanced 16k/2k/2k metadata; no model or Q evaluation."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import json
from pathlib import Path
import signal
import sys
import time
import torch
from datasets.paths import dataset_paths, resolved_data_paths
from datasets.reference_layouts import VERSION, TASK, REGISTRY, inspect_shard, digest, registry
from .io import HERE, ROOT, save, sha, exclusive
from .balancing import common_quotas, select_records, monitor_records, rank
from .preparation_workers import initialize_cpu, inspect_candidate, compatible_identity

DEFAULT = HERE/'runs/20261019_balanced_v5_seed20261019'


def metadata_identity(seed):
    names = ['datasets/reference_layouts.py', 'experiments/E16_reference_curriculum/prepare.py',
             'experiments/E16_reference_curriculum/balancing.py', 'experiments/E16_reference_curriculum/io.py',
             'experiments/E16_reference_curriculum/preparation_workers.py']
    return {'dataset_version': VERSION, 'task_version': TASK, 'seed': seed,
            'frozen_original_hashes': {n: sha(ROOT/n) for n in
                ('protocol/split_v1.json', 'protocol/workset_v2.json', 'protocol/derived_workset_v3.json')},
            'paths_sha256': sha(ROOT/'protocol/paths.json'), 'torch_version': str(torch.__version__),
            'preparation_sources': {n: sha(ROOT/n) for n in names}}


def create_registry(scans, *, identity, seed, pilot=False):
    candidates = {r: [] for r in ('train', 'dev', 'local_test')}
    raw_hist = {r: Counter() for r in candidates}
    rejects = {r: Counter() for r in candidates}
    # Conservatively exclude the entire old v2 test panel: its known exposed
    # prefix and possibly in-flight member cannot enter the new test set.
    v2 = json.loads((ROOT/'protocol/workset_v2.json').read_text(encoding='utf-8'))
    historical_test = {(entry['shard'], row) for entry in v2['members']['local_test']
                       for row in entry['sample_indices']}
    excluded_test = Counter()
    for scan in scans:
        role = scan['parent']
        for row, n in scan['accepted']:
            if role == 'local_test' and (scan['shard'], row) in historical_test:
                excluded_test[n] += 1
                continue
            candidates[role].append({'member': [scan['shard'], row], 'n': n,
                                    'case_id': f'{role}:{scan["shard"]}:{row}'})
        raw_hist[role].update({int(n): v for n, v in scan['raw_n_counts'].items()})
        rejects[role].update(scan['rejections'])
    available = {r: Counter(x['n'] for x in rows) for r, rows in candidates.items()}
    quotas, caps = common_quotas(available, seed=seed)
    sets = {r: select_records(rows, quotas, r, seed=seed, factor=8 if r == 'train' else 1)
            for r, rows in candidates.items()}
    sets['train_monitor500'] = monitor_records(sets['train'], seed=seed, tag='train_monitor500')
    sets['dev_monitor500'] = monitor_records(sets['dev'], seed=seed, tag='dev_monitor500')
    selected_shards = {x['member'][0] for rows in sets.values() for x in rows}
    data = {'dataset_version': VERSION, 'task_version': TASK, 'seed': seed, 'pilot': pilot,
            'identity': identity, 'frozen_original_hashes': identity['frozen_original_hashes'],
            'targets': {'train': 16000, 'dev': 2000, 'local_test': 2000},
            'partition_rule': 'Original split_v1 whole-shard boundaries unchanged; independently rerank all eligible rows in each pool',
            'balance_rule': 'Actual block count n; max-min q_n with cap=min(floor(train_n/8),dev_n,test_n); exact train:dev:test=8:1:1 in each n',
            'label_source': 'Original reference centers and log aspect ratios, no teacher or quality-based selection',
            'historical_test_exclusion': {'rule': 'All 2500 original workset_v2 local_test members excluded, including its exposed prefix and possible in-flight cases',
                                          'candidate_count': len(historical_test), 'compatible_excluded_n_counts': dict(excluded_test)},
            'test_status': 'Membership/public-compatibility preparation only; labels, model inference and scoring sealed',
            'selection_uses_Q': False, 'selection_uses_free_labels': False,
            'candidate_n_counts': {r: dict(c) for r, c in available.items()},
            'raw_n_counts': {r: dict(c) for r, c in raw_hist.items()},
            'rejections': {r: dict(c) for r, c in rejects.items()},
            'common_capacity': caps, 'dev_test_quotas': quotas,
            'unsupported_n': [n for n, cap in caps.items() if cap == 0],
            'sets': {r: {'count': len(rows), 'n_counts': dict(Counter(x['n'] for x in rows)), 'records': rows}
                     for r, rows in sets.items()},
            'source_shards': {s['shard']: {'size': s['size'], 'mtime_ns': s['mtime_ns']}
                              for s in scans if s['shard'] in selected_shards}}
    data['manifest_sha256'] = digest(data)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=DEFAULT)
    ap.add_argument('--seed', type=int, default=20261019)
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = ap.parse_args(); args.output = args.output.resolve()
    if not args.output.is_relative_to(HERE/'runs'):
        raise ValueError('Preparation outputs must stay in E16/runs')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    ident = metadata_identity(args.seed)
    split = json.loads((ROOT/'protocol/split_v1.json').read_text(encoding='utf-8'))
    shards = [(role, s) for role in ('train', 'dev', 'local_test') for s in split['shards'][role]]
    shards.sort(key=lambda x: (rank(args.seed, 'scan_order', x[1]), x))
    begin = time.perf_counter(); root = dataset_paths()['training_root']
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive(args.output/'prepare.lock'), exclusive(REGISTRY.with_suffix('.lock')):
        if REGISTRY.exists():
            existing = registry()
            if not compatible_identity(existing['identity'], ident, 'candidate_scan'):
                raise ValueError('v5 already frozen with different seed/source; use an explicit new version')
            print('Registry already frozen; reusing:', REGISTRY, flush=True)
            return
        identity_file = args.output/'identity.json'
        previous_identity = json.loads(identity_file.read_text(encoding='utf-8')) if identity_file.exists() else None
        if previous_identity is not None and not compatible_identity(previous_identity, ident, 'candidate_scan'):
            raise ValueError('Preparation identity changed; do not mix interrupted scans')
        if previous_identity is None:
            save(identity_file, ident)
        scan_dir = args.output/'shards'; scan_dir.mkdir(exist_ok=True)
        calls_path = args.output/'invocations.json'
        calls = json.loads(calls_path.read_text(encoding='utf-8')) if calls_path.exists() else []
        calls.append({'command': [sys.executable, *sys.argv], 'workers': args.workers,
                      'parallelism': 'spawn processes', 'execution_identity': ident,
                      'torch_threads': torch.get_num_threads(), 'library_threads_per_worker': 1,
                      'data_paths': resolved_data_paths(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        save(calls_path, calls)
        scans, todo = [], []
        for role, shard in shards:
            path = scan_dir/(shard.replace('/', '__')+'.json')
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                stat = (root/shard).stat()
                if record['shard'] != shard or record['parent'] != role or (record['size'], record['mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError('Completed candidate source changed: '+shard)
                scans.append(record)
            else:
                todo.append((role, shard, path))
        requested = [False]
        handler = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        stopped = lambda: requested[0] or (args.output/'STOP').exists()
        prior = json.loads((args.output/'progress.json').read_text(encoding='utf-8')).get('elapsed_seconds', 0.) if (args.output/'progress.json').exists() else 0.
        def progress(status):
            save(args.output/'progress.json', {'status': status, 'completed_shards': len(scans), 'total_shards': len(shards),
                'elapsed_seconds': prior+time.perf_counter()-begin, 'workers': args.workers, 'parallelism': 'spawn processes',
                'test_public_metadata_only': True, 'test_scored': False})
        try:
            pending_iter = iter(todo)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                     initializer=initialize_cpu) as pool:
                pending = {}
                def submit_one():
                    if stopped():
                        return
                    item = next(pending_iter, None)
                    if item:
                        role, shard, path = item
                        pending[pool.submit(inspect_candidate, shard, role)] = path
                for _ in range(args.workers):
                    submit_one()
                while pending:
                    done, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
                    for future in done:
                        path = pending.pop(future); record = future.result()
                        save(path, record); scans.append(record)
                        progress('scanning')
                        if len(scans) % 25 == 0:
                            print(f'Public candidate scan: {len(scans)}/{len(shards)} shards', flush=True)
                        submit_one()
            if stopped():
                progress('user_stop'); print('Stopped; remove STOP and rerun the same command to resume', flush=True); return
            if len(scans) != len(shards):
                raise ValueError('Incomplete candidate scan')
            data = create_registry(sorted(scans, key=lambda s: s['shard']), identity=ident, seed=args.seed)
            if metadata_identity(args.seed) != ident:
                raise ValueError('Sources or frozen originals changed during preparation')
            # The only registry is written after all counts and source-boundary
            # checks succeed. Re-entry cannot choose different members.
            save(REGISTRY, data)
            registry()
            prior_sets = json.loads((ROOT/'protocol/derived_workset_v3.json').read_text(encoding='utf-8'))['sets']
            old_overlap = {r: len({tuple(x['member']) for x in data['sets'][r]['records']} &
                                   {tuple(m) for m in prior_sets[r]['members']}) for r in ('train', 'dev')}
            save(args.output/'results.json', {'status': 'complete', 'registry': str(REGISTRY),
                'registry_sha256': sha(REGISTRY), 'elapsed_seconds': prior+time.perf_counter()-begin,
                'counts': {r: entry['count'] for r, entry in data['sets'].items()},
                'n_counts': {r: entry['n_counts'] for r, entry in data['sets'].items()},
                'old_v3_member_overlap': old_overlap, 'old_overlap_is_allowed': True,
                'test_public_metadata_only': True, 'test_scored': False,
            'historical_test_exposure': 'Entire v2 test panel excluded conservatively; other historical exposures remain disclosed in docs/protocol.md; no claim that randomization erases exposure'})
            progress('complete')
            print('Frozen new members:', REGISTRY, flush=True)
            print(json.dumps({r: e['n_counts'] for r, e in data['sets'].items()}, sort_keys=True), flush=True)
        finally:
            signal.signal(signal.SIGINT, handler)


if __name__ == '__main__':
    main()
