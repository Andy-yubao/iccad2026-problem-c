"""Evaluate frozen E169 Ref080 with the accelerated repair on the original dev2000."""

import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from datasets.random_reference import RandomReference
from experiments.E165_joint_l0.evaluation.repair.canonical_native import warmup
from experiments.E165_joint_l0.evaluation.run import case_from_item, public_graph, repair_inputs
from experiments.E165_joint_l0.evaluation.solver import BUDGET
from experiments.E169_s2_progressive_unfreeze.batches import decode
from experiments.E169_s2_progressive_unfreeze.io import ROOT, exclusive, save, sha
from experiments.E169_s2_progressive_unfreeze.model import GeometryModel
from research.quality_log import summary as quality_line
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from scoring.research_no_mib import POLICY, evaluate_case
from .benchmark import write
from .optimizer import optimize


OWNER = ROOT / 'experiments/E169_s2_progressive_unfreeze'
ARM = 'e169_ref080_best33_accelerated_repair'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def initialize(checkpoint, expected_hash, cases, hashes):
    global dataset, model, case_dir, payload_hashes, native_limits
    native_limits = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = RandomReference('dev')
    case_dir, payload_hashes = cases, hashes
    if sha(checkpoint) != expected_hash:
        raise ValueError('E169 frozen checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = GeometryModel(**state['model_config']).eval()
    model.load_state_dict(state['model'], strict=True)
    official_evaluator()
    warmup()
    # Warm up the native signatures on a synthetic layout outside panel timings.
    draft = np.array([[0., 0., 2., 2.], [2., 0., 2., 2.]])
    inputs = (2, np.array([4., 4.]), np.array([[0., 1., 1.]]),
              np.empty((0, 3)), np.empty((0, 2)), np.zeros((2, 5), dtype=np.int64),
              np.zeros((2, 4)))
    optimize(draft, inputs)


def run_case(index):
    record = dataset.records[dataset.indices[index]]
    if sha(dataset.folder / record['payload']) != payload_hashes[record['id']]:
        raise ValueError(f'Dev payload changed at case {index}')
    item = dataset[index]
    inputs = item['solver_inputs']
    n = inputs[0]
    started = time.perf_counter()
    cpu_started = time.process_time()
    adjusted = repair_inputs(inputs)
    with torch.inference_mode():
        graph = public_graph(adjusted)
        draft = (decode(model(graph)[-1], graph)[0, :n] * graph['scale'][0]).numpy().astype(float)
    if not np.isfinite(draft).all():
        raise ValueError(f'Nonfinite draft at case {index}')
    draft_seconds = time.perf_counter() - started
    final, info = optimize(draft, adjusted)
    checked = independent_hard_checks(final.tolist(), adjusted[1], adjusted[5], adjusted[6])
    complete = time.perf_counter() - started
    complete_cpu = time.process_time() - cpu_started
    if not checked['is_feasible'] or (final[:, :2] < 0).any():
        raise ValueError(f'Independent hard legality failed at case {index}: {checked}')

    metrics = evaluate_case(case_from_item(item), final, complete)['metrics']
    if not metrics['is_feasible']:
        raise ValueError(f'Official scoring rejected case {index}')
    row = dict(index=index, member=item['member'], n=n, q=metrics['cost'], hard_legal=True,
               grouping=metrics['grouping_violations'], boundary=metrics['boundary_violations'],
               violation_ratio=metrics['violations_relative'], area_gap=metrics['area_gap'],
               hpwl_gap=metrics['hpwl_gap'], draft_seconds=draft_seconds,
               complete_seconds=complete, complete_cpu_seconds=complete_cpu,
               repair_seconds=info['seconds'], row_retained=info['baseline']['row_retained'],
               refinement=info['refinement'])
    write(case_dir / f'{index:04d}.npz', draft, final, row)
    return index


def summarize(rows):
    q = np.array([row['q'] for row in rows])
    seconds = np.array([row['complete_seconds'] for row in rows])
    weighted = official_evaluator().compute_total_score(q.tolist(), [row['n'] for row in rows])
    mean = float(q.mean())
    return dict(count=len(rows), weighted_Q=weighted, mean_Q=mean, mixed_Q=(weighted + mean) / 2,
                p95_Q=float(np.percentile(q, 95)), max_Q=float(q.max()),
                hard_legal=sum(row['hard_legal'] for row in rows),
                grouping_violations=sum(row['grouping'] for row in rows),
                boundary_violations=sum(row['boundary'] for row in rows),
                grouping_satisfied=sum(row['grouping'] == 0 for row in rows),
                boundary_satisfied=sum(row['boundary'] == 0 for row in rows),
                all_soft_satisfied=sum(row['grouping'] == row['boundary'] == 0 for row in rows),
                mean_soft_violation_ratio=float(np.mean([row['violation_ratio'] for row in rows])),
                mean_hpwl_gap=float(np.mean([row['hpwl_gap'] for row in rows])),
                mean_area_gap=float(np.mean([row['area_gap'] for row in rows])),
                mean_complete_seconds=float(seconds.mean()),
                p95_complete_seconds=float(np.percentile(seconds, 95)),
                max_complete_seconds=float(seconds.max()), over250ms=int((seconds > .25).sum()),
                mean_complete_cpu_seconds=float(np.mean([row['complete_cpu_seconds'] for row in rows])),
                mean_repair_seconds=float(np.mean([row['repair_seconds'] for row in rows])),
                main_row_retained=sum(row['row_retained'] for row in rows),
                refinement_cases_changed=sum(bool(row['refinement']['soft_moves'] or
                                                  row['refinement']['legal_seed_accepted'] or
                                                  row['refinement']['anchor_seed_accepted']) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.workers < 1 or out.parent != OWNER / 'evaluation' or out.name == 'output':
        parser.error('Use positive workers and a dedicated E169 evaluation directory')
    if out.exists() != args.resume:
        parser.error('Use a new output directory or explicitly resume an existing run')
    started = time.perf_counter()
    frozen = read(OWNER / 'frozen_e169_s2.json')
    gate = read(OWNER / 'plan.json')
    panel_path = ROOT / 'protocol/final_evaluation_no_mib_dev2000_v1.json'
    panel = read(panel_path)
    dev = RandomReference('dev')
    members = [dev.records[i]['member'] for i in dev.indices]
    panel_members = [row['member'] for row in panel['members']]
    monitor = set(panel['subsets']['monitor500'])
    extension = set(panel['subsets']['extension1500'])
    if (members != panel_members or len(members) != 2000 or len(monitor) != 500 or
            len(extension) != 1500 or monitor & extension or monitor | extension != set(dev.indices)):
        raise ValueError('Frozen dev2000 members changed')
    ledger = dev.folder / 'checksums.jsonl'
    if (sha(ledger) != gate['payload_checksums_sha256'] or
            sha(dev.folder / 'report.json') != gate['dataset_report_sha256']):
        raise ValueError('Dataset identity changed')
    hashes = {row['id']: row['payload_sha256'] for line in ledger.read_text(encoding='utf-8').splitlines()
              for row in json.loads(line)['records'] if row['role'] == 'dev'}
    checkpoint = ROOT / frozen['checkpoint']
    expected_hash = frozen['checkpoint_sha256']
    if sha(checkpoint) != expected_hash or len(hashes) != len(members):
        raise ValueError('Checkpoint or dev checksum count changed')
    sources = [Path(__file__), panel_path, *(ROOT / 'research/postprocess_tail').glob('*.py'),
               *OWNER.glob('*.py'), *(ROOT / 'experiments/E165_joint_l0/evaluation').rglob('*.py'),
               *(ROOT / 'experiments/E165_joint_l0' / name for name in
                 ('features.py', 'ports.py', 'role_features.py'))]
    identity = dict(arm=ARM, checkpoint=frozen['checkpoint'], checkpoint_sha256=expected_hash,
                    best_epoch=frozen['best_epoch'], payload_checksums_sha256=sha(ledger),
                    budget=BUDGET['revision'], scoring_policy=POLICY, runtime_multiplier=1.0,
                    workers=args.workers, device='cpu', test_read=False,
                    libraries={key: version(key) for key in ('torch', 'numpy', 'clarabel', 'numba', 'shapely')},
                    sources={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(sources))})
    with exclusive(out / 'evaluation.lock'):
        marker = out / 'frozen.json'
        if marker.exists() and read(marker) != identity:
            raise ValueError('Evaluation identity changed')
        save(marker, identity)
        cases = out / 'cases'
        cases.mkdir(exist_ok=True)
        pending = [i for i in range(len(members)) if not (cases / f'{i:04d}.npz').exists()]
        done = len(members) - len(pending)
        initargs = checkpoint, expected_hash, cases, hashes

        def progress(status):
            save(out / 'state.json', dict(status=status, completed=done, total=len(members),
                                          pid=os.getpid(), elapsed_seconds=time.perf_counter() - started))

        progress('initializing')
        try:
            if pending:
                # Populate the native cache before spawning the Windows workers.
                initialize(*initargs)
                progress('running')
                with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                         initializer=initialize, initargs=initargs) as pool:
                    futures = {pool.submit(run_case, i): i for i in pending}
                    for future in as_completed(futures):
                        future.result()
                        done += 1
                        if done == 1 or done % 25 == 0:
                            progress('running')
                            print(f'{done}/{len(members)}', flush=True)
            rows = []
            for index, member in enumerate(members):
                with np.load(cases / f'{index:04d}.npz', allow_pickle=False) as case:
                    row = json.loads(case['metrics'].item())
                if row['index'] != index or row['member'] != member:
                    raise ValueError(f'Saved case identity differs at index {index}')
                rows.append(row)
            subsets = dict(monitor500=summarize([row for row in rows if row['index'] in monitor]),
                           extension1500=summarize([row for row in rows if row['index'] not in monitor]),
                           full2000=summarize(rows))
            if sha(checkpoint) != expected_hash:
                raise ValueError('Checkpoint changed during evaluation')
            result = dict(status='complete', arm=ARM, checkpoint_sha256=expected_hash,
                          best_epoch=frozen['best_epoch'], budget=BUDGET['revision'], scoring_policy=POLICY,
                          runtime_multiplier=1.0, test_read=False, workers=args.workers,
                          elapsed_seconds=time.perf_counter() - started,
                          timing_scope='Concurrent CPU workers; per-case wall includes input copy, inference, full repair and independent hard check; excludes queue, disk and scoring.',
                          **subsets['full2000'])
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
