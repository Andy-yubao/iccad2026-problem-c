"""Run frozen E169 best33 with a GPU draft dispatcher and the accelerated CPU repair."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import time

from ..evaluate_e169 import (BUDGET, OWNER, POLICY, ROOT, RandomReference, case_from_item,
                            evaluate_case, exclusive, independent_hard_checks, official_evaluator,
                            optimize, quality_line, read, repair_inputs, save, sha, summarize,
                            threadpool_limits, warmup, write)
import numpy as np
import torch

from .predictor import Predictor


ARM = 'e169_ref080_best33_gpu_graph_accelerated_repair'
ORIGINAL = OWNER / 'evaluation/accelerated_dev2000_v1'


def initialize_worker():
    global native_limits
    native_limits = threadpool_limits(limits=1)
    official_evaluator()
    warmup()
    draft = np.array([[0., 0., 2., 2.], [2., 0., 2., 2.]])
    inputs = (2, np.array([4., 4.]), np.array([[0., 1., 1.]]), np.empty((0, 3)),
              np.empty((0, 2)), np.zeros((2, 5), dtype=np.int64), np.zeros((2, 4)))
    optimize(draft, inputs)


def repair(index, item, inputs, draft, elapsed):
    start = time.perf_counter()
    cpu_start = time.process_time()
    final, info = optimize(draft, inputs)
    checked = independent_hard_checks(final.tolist(), inputs[1], inputs[5], inputs[6])
    complete = elapsed['draft_seconds'] + time.perf_counter() - start
    complete_cpu = elapsed['draft_cpu_seconds'] + time.process_time() - cpu_start
    if not checked['is_feasible'] or (final[:, :2] < 0).any():
        raise ValueError(f'Independent hard legality failed at case {index}: {checked}')
    metrics = evaluate_case(case_from_item(item), final, complete)['metrics']
    if not metrics['is_feasible']:
        raise ValueError(f'Official scoring rejected case {index}')
    row = dict(index=index, member=item['member'], n=inputs[0], q=metrics['cost'], hard_legal=True,
               grouping=metrics['grouping_violations'], boundary=metrics['boundary_violations'],
               violation_ratio=metrics['violations_relative'], area_gap=metrics['area_gap'],
               hpwl_gap=metrics['hpwl_gap'], complete_seconds=complete,
               complete_cpu_seconds=complete_cpu, repair_seconds=info['seconds'],
               row_retained=info['baseline']['row_retained'], refinement=info['refinement'],
               **elapsed)
    return draft, final, row


def load_case(path, index, member):
    with np.load(path, allow_pickle=False) as case:
        row = json.loads(case['metrics'].item())
        if row['index'] != index or row['member'] != member:
            raise ValueError(f'Wrong saved case at index {index}')
        return case['draft'], case['final'], row


def paired(cases, members):
    rows = []
    original = []
    equal_drafts = equal_final = 0
    draft_errors = []
    for i, member in enumerate(members):
        old_draft, old_final, old = load_case(ORIGINAL / 'cases' / f'{i:04d}.npz', i, member)
        new_draft, new_final, new = load_case(cases / f'{i:04d}.npz', i, member)
        equal_drafts += np.array_equal(old_draft, new_draft)
        equal_final += np.array_equal(old_final, new_final)
        draft_errors.append(float(np.abs(old_draft - new_draft).max()))
        original.append(old)
        rows.append(new)
    deltas = np.array([new['q'] - old['q'] for old, new in zip(original, rows)])
    quality = dict(equal_drafts=int(equal_drafts), equal_final_layouts=int(equal_final),
                   max_abs_draft_error=max(draft_errors),
                   p95_case_max_abs_draft_error=float(np.percentile(draft_errors, 95)),
                   q_improved=int((deltas < -1e-9).sum()), q_unchanged=int((np.abs(deltas) <= 1e-9).sum()),
                   q_regressed=int((deltas > 1e-9).sum()), largest_q_increase=float(deltas.max()),
                   largest_q_drop=float(deltas.min()),
                   worst_increase_cases=sorted(range(len(deltas)), key=lambda i: -deltas[i])[:5])
    return original, rows, quality


def timing(rows):
    names = ('draft_seconds', 'graph_seconds', 'forward_seconds', 'repair_seconds', 'complete_seconds')
    result = {}
    for name in names:
        if name in rows[0]:
            values = np.array([row[name] * 1000 for row in rows])
            result[name.removesuffix('_seconds')] = dict(mean_ms=float(values.mean()),
                                                        p95_ms=float(np.percentile(values, 95)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.workers < 1 or out.parent != OWNER / 'evaluation' or out in (ORIGINAL, OWNER / 'evaluation/output'):
        parser.error('Use positive workers and a new E169 evaluation directory')
    if out.exists() != args.resume:
        parser.error('Use a new output directory or explicitly resume an existing run')

    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen = read(OWNER / 'frozen_e169_s2.json')
    checkpoint = ROOT / frozen['checkpoint']
    checkpoint_hash = frozen['checkpoint_sha256']
    if sha(checkpoint) != checkpoint_hash:
        raise ValueError('Frozen E169 checkpoint changed')
    archived = read(ORIGINAL / 'frozen.json')
    if archived['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Original CPU evaluation checkpoint differs')
    panel_path = ROOT / 'protocol/final_evaluation_no_mib_dev2000_v1.json'
    panel = read(panel_path)
    dev = RandomReference('dev')
    members = [dev.records[i]['member'] for i in dev.indices]
    if len(members) != 2000 or members != [row['member'] for row in panel['members']]:
        raise ValueError('Registered dev2000 member order changed')
    ledger = dev.folder / 'checksums.jsonl'
    if sha(ledger) != archived['payload_checksums_sha256']:
        raise ValueError('Original dev2000 payload ledger changed')
    payload_hashes = {row['id']: row['payload_sha256']
                      for line in ledger.read_text(encoding='utf-8').splitlines()
                      for row in json.loads(line)['records'] if row['role'] == 'dev'}
    if len(payload_hashes) != 2000:
        raise ValueError('Missing dev checksums')
    sources = set(archived['sources'])
    sources.update(path.relative_to(ROOT).as_posix() for path in Path(__file__).parent.glob('*.py'))
    identity = dict(arm=ARM, checkpoint=frozen['checkpoint'], checkpoint_sha256=checkpoint_hash,
                    best_epoch=frozen['best_epoch'], payload_checksums_sha256=sha(ledger),
                    budget=BUDGET['revision'], scoring_policy=POLICY, test_read=False, workers=args.workers,
                    device=torch.cuda.get_device_name(), inference='single-case FP32 CUDA Graph; TF32 disabled',
                    libraries={key: version(key) for key in ('torch', 'numpy', 'clarabel', 'numba', 'shapely')},
                    sources={name: sha(ROOT / name) for name in sorted(sources)})
    with exclusive(out / 'evaluation.lock'):
        marker = out / 'frozen.json'
        if marker.exists() and read(marker) != identity:
            raise ValueError('Evaluation identity changed')
        save(marker, identity)
        cases = out / 'cases'
        cases.mkdir(exist_ok=True)
        remaining = [i for i in range(len(members)) if not (cases / f'{i:04d}.npz').exists()]
        done = len(members) - len(remaining)
        started = time.perf_counter()

        def progress(status):
            save(out / 'state.json', dict(status=status, completed=done, total=len(members), pid=os.getpid(),
                                          elapsed_seconds=time.perf_counter() - started))

        progress('running')
        try:
            if remaining:
                predictor = Predictor(checkpoint, checkpoint_hash)
                with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                         initializer=initialize_worker) as executor:
                    pending = {}

                    def collect():
                        nonlocal done
                        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in ready:
                            index = pending.pop(future)
                            draft, final, row = future.result()
                            write(cases / f'{index:04d}.npz', draft, final, row)
                            done += 1
                            if done == 1 or done % 25 == 0:
                                progress('running')
                                print(f'{done}/{len(members)}', flush=True)

                    for i in remaining:
                        record = dev.records[dev.indices[i]]
                        if sha(dev.folder / record['payload']) != payload_hashes[record['id']]:
                            raise ValueError(f'Dev payload changed at case {i}')
                        item = dev[i]
                        start = time.perf_counter()
                        cpu_start = time.process_time()
                        inputs = repair_inputs(item['solver_inputs'])
                        draft, elapsed = predictor.predict(inputs)
                        elapsed['draft_seconds'] = time.perf_counter() - start - elapsed['capture_seconds']
                        elapsed['draft_cpu_seconds'] = max(
                            0., time.process_time() - cpu_start - elapsed['capture_cpu_seconds'])
                        pending[executor.submit(repair, i, item, inputs, draft, elapsed)] = i
                        if len(pending) >= 2 * args.workers:
                            collect()
                    while pending:
                        collect()
            original, rows, numerical = paired(cases, members)
            monitor = set(panel['subsets']['monitor500'])
            subsets = dict(monitor500=summarize([row for row in rows if row['index'] in monitor]),
                           extension1500=summarize([row for row in rows if row['index'] not in monitor]),
                           full2000=summarize(rows))
            result = dict(status='complete', arm=ARM, checkpoint_sha256=checkpoint_hash,
                          best_epoch=frozen['best_epoch'], budget=BUDGET['revision'], scoring_policy=POLICY,
                          runtime_multiplier=1.0, test_read=False, workers=args.workers,
                          elapsed_seconds=time.perf_counter() - started,
                          capture_seconds=sum(row['capture_seconds'] for row in rows),
                          timing_scope='Sum of per-case GPU-dispatch and CPU-repair service times, including '
                                       'input copy, graph construction, transfers, synchronization and independent '
                                       'hard check; dispatch wait, disk, model loading, capture and scoring separate.',
                          **subsets['full2000'])
            before = summarize(original)
            keys = ('mixed_Q', 'p95_Q', 'max_Q', 'hard_legal', 'grouping_violations',
                    'boundary_violations', 'all_soft_satisfied')
            comparison = dict(original_cpu={key: before[key] for key in keys},
                              cuda_graph={key: result[key] for key in keys},
                              difference={key: result[key] - before[key] for key in keys},
                              original_timing=timing(original), cuda_graph_timing=timing(rows),
                              numerical=numerical)
            save(out / 'comparison.json', comparison)
            save(out / 'subsets.json', subsets)
            save(out / 'results.json', result)
            line = quality_line(result, (out / 'results.json').relative_to(ROOT).as_posix())
            (out / 'quality.log').write_text(line + '\n', encoding='utf-8')
            print(line, flush=True)
            progress('complete')
        except BaseException as error:
            save(out / 'error.json', dict(error=repr(error), completed=done))
            progress('error')
            raise


if __name__ == '__main__':
    main()
