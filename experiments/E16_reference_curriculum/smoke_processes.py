"""Five known train/dev cases: four spawned workers and cache equivalence."""
import argparse
import json
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output = args.output.resolve()
    here = Path(__file__).resolve().parent
    if not args.output.is_relative_to(here/'runs'):
        raise ValueError('Smoke output must stay in E16/runs')
    args.output.mkdir(parents=True, exist_ok=False)
    pilot = here/'runs/20261019_reference_smoke5_seed0_r2/pilot_registry.json'
    os.environ['E16_REGISTRY'] = str(pilot)
    import numpy as np
    import torch
    from datasets.reference_layouts import ReferenceLayouts, inspect_shard
    from .preparation_workers import initialize_cpu, inspect_candidate
    from .cache import build_role, PackedRows
    from .data import prepare
    from .io import save
    torch.set_num_threads(1)
    started = time.perf_counter()
    meta = json.loads(pilot.read_text(encoding='utf-8'))
    context = mp.get_context('spawn'); ready = context.Queue()
    jobs = [(r['member'][0], role, [r['member'][1]]) for role in ('train', 'dev') for r in meta['sets'][role]['records']]
    # Only five unique, previously checked layouts; no test data or inference.
    with ProcessPoolExecutor(max_workers=4, mp_context=context, initializer=initialize_cpu, initargs=(ready,)) as pool:
        futures = [pool.submit(inspect_candidate, *job) for job in jobs]
        actual = [f.result(timeout=60) for f in futures]
        workers = [ready.get(timeout=10) for _ in range(4)]
    assert len({w['pid'] for w in workers}) == 4
    assert all(w['torch_threads'] == w['interop_threads'] == 1 for w in workers)
    for job, record in zip(jobs, actual):
        expected = inspect_shard(job[0], job[1], sample_rows=job[2])
        assert {k: v for k,v in record.items() if k != 'execution'} == expected
    cache_root = args.output/'cache'
    row_equivalence = 0
    for role in ('train', 'dev'):
        ids = list(range(len(meta['sets'][role]['records'])))
        # Stop after worker parts have been saved, before final merge. Resume
        # must consume those parts without scheduling their preprocessing again.
        stop_after_parts = lambda: any((cache_root/role/'parts').glob('*.npz'))
        assert build_role(role, cache_root/role, indices=ids, workers=4, chunk_size=1,
                          should_stop=stop_after_parts) is False
        assert build_role(role, cache_root/role, indices=ids, workers=4, chunk_size=1)
        merged = json.loads((cache_root/role/'manifest.json').read_text(encoding='utf-8'))
        assert merged['execution']['reused_rows'] == len(ids)
        cached = PackedRows(role, cache_root=cache_root, allow_pilot=True)
        dataset = ReferenceLayouts(role)
        for i in ids:
            a, b = prepare(dataset[i]), cached[i]
            assert a['member'] == b['member']
            for key in a['graph']:
                assert torch.equal(a['graph'][key], b['graph'][key]), key
            for key in ('geometry','b2b','p2b','pins'):
                assert torch.equal(a['target'][key], b['target'][key]), key
            row_equivalence += 1
        # Re-entry must reuse the finished columns, without spawning work or
        # replacing any data files. Source/data identity is checked first.
        before = {p.name: p.stat().st_mtime_ns for p in (cache_root/role).glob('*.npy')}
        assert build_role(role, cache_root/role, indices=ids, workers=4, chunk_size=1)
        assert before == {p.name: p.stat().st_mtime_ns for p in (cache_root/role).glob('*.npy')}
    save(args.output/'results.json', {'passed': True, 'unique_real_cases': 5,
        'actual_spawn_workers': workers, 'worker_records': [r['execution'] for r in actual],
        'public_scan_identical': True, 'cache_rows_identical': row_equivalence,
        'completed_cache_reused_without_rewrite': True, 'test_read': False,
        'partial_parts_reused_without_reprocessing': True,
        'formal_preparation_restarted': False, 'elapsed_seconds': time.perf_counter()-started})
    print('PASS: 4 distinct single-thread worker processes; 5 cases identical; finished caches reused', flush=True)


if __name__ == '__main__':
    main()
