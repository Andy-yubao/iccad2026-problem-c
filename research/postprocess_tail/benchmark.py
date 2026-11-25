"""Paired dev2000 benchmark with frozen E165 predictions and independent scoring."""
import os

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from datasets.random_reference import RandomReference
from experiments.E165_joint_l0.evaluation.run import public_graph, case_from_item, repair_inputs
from experiments.E165_joint_l0.evaluation.solver import optimize
from experiments.E165_joint_l0.evaluation.repair.canonical_native import warmup
from experiments.E165_joint_l0.io import sha
from experiments.E165_joint_l0.batches import decode
from experiments.E165_joint_l0.model import GeometryModel
from scoring.research_no_mib import evaluate_case
from scoring.official import official_evaluator
from .optimizer import refine

ROOT=Path(__file__).resolve().parents[2]
PLAN=json.loads((ROOT/'experiments/E165_joint_l0/evaluation/plan.json').read_text())
FROZEN=json.loads((ROOT/'experiments/E165_joint_l0/evaluation/output/frozen.json').read_text())

def init():
    global dataset, hashes, model, out
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset=RandomReference('dev')
    if [dataset.records[i]['member'] for i in dataset.indices] != PLAN['members']:
        raise ValueError('Frozen dev2000 members changed')
    ledger=dataset.folder/'checksums.jsonl'
    if sha(ledger) != PLAN['payload_checksums_sha256']:
        raise ValueError('Frozen dataset checksum ledger changed')
    hashes={r['id']:r['payload_sha256'] for line in ledger.read_text().splitlines()
            for r in json.loads(line)['records'] if r['role']=='dev'}
    if args.original is None:
        if sha(args.checkpoint) != FROZEN['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model = GeometryModel(**state['model_config']).eval()
        model.load_state_dict(state['model'], strict=True)
    if not official_evaluator().SHAPELY_AVAILABLE:
        raise RuntimeError('Shapely is required for grouping evaluation')
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    warmup()

def solve(i):
    path=out/f'{i:04d}.npz'
    if path.exists():
        return i
    ref=dataset.records[i]
    if sha(dataset.folder / ref['payload']) != hashes[ref['id']]:
        raise ValueError(f'Dev payload changed for case {i}')
    item=dataset[i]
    case=case_from_item(item)
    inputs=item['solver_inputs']
    adjusted=repair_inputs(inputs)
    if args.original is not None:
        with np.load(args.original/f'{i:04d}.npz', allow_pickle=False) as cached:
            draft, original = cached['draft'], cached['final']
            old = json.loads(cached['metrics'].item())
        final, info = refine(draft, adjusted, original, legal_main=old['legal_main'])
        result = metrics(case, final, info['seconds'], i)
        result.update(info)
        result.update(old_seconds=old['seconds'], draft_seconds=old['draft_seconds'])
        if (result['q'] > old['q'] + 1e-9 or result['group'] > old['group']
                or result['boundary'] > old['boundary']):
            raise AssertionError((i, old, result))
        write(path, draft, final, result)
        return i
    with torch.inference_mode():
        t=time.perf_counter()
        graph=public_graph(inputs)
        state=model(graph)[-1]
        draft=(decode(state,graph)[0,:inputs[0]]*graph['scale'][0]).numpy().astype(float)
        draft_time=time.perf_counter()-t
    if not np.isfinite(draft).all():
        raise ValueError(f'Nonfinite draft for case {i}')
    final,info=optimize(draft,adjusted)
    phases=[v['phase'] for v in info['trace'] if v.get('legal')]
    result=metrics(case, final, info['seconds'], i)
    result.update(draft_seconds=draft_time, legal_main='shape' in phases, phases=phases,
        work=info['work'], row_retained=info['row_retained'], proxy=info['proxy'])
    write(path, draft, final, result)
    return i

def metrics(case, layout, seconds, i):
    m = evaluate_case(case, layout, seconds)['metrics']
    if not m['is_feasible']:
        raise AssertionError(f'Hard legality failed for case {i}')
    return dict(i=i, n=case.block_count, q=m['cost'], group=m['grouping_violations'],
                boundary=m['boundary_violations'], nsoft=m['max_possible_violations'],
                area=m['bbox_area'], wire=m['hpwl_total'], seconds=seconds)

def write(path, draft, final, result):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,draft=draft,final=final,metrics=json.dumps(result))
    tmp.replace(path)

def main():
    global args
    parser=argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--original', type=Path)
    parser.add_argument('--monitor-only', action='store_true')
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers', type=int, default=1 if os.name == 'nt' else 7)
    args = parser.parse_args()
    if args.workers < 1 or (os.name == 'nt' and args.workers != 1):
        parser.error('Use one or more workers on Linux and one on Windows')
    if args.original is not None and args.original.resolve() == args.out.resolve():
        parser.error('Use separate directories for original and candidate layouts')
    init()
    print('Verified panel and payload ledger',flush=True)
    monitor = PLAN['monitor500_indices']
    selected = set(monitor)
    order = args.indices or (monitor if args.monitor_only else
                             monitor + [i for i in range(2000) if i not in selected])
    if len(set(order)) != len(order) or any(i < 0 or i >= 2000 for i in order):
        parser.error('Case indices must be unique and in [0, 1999]')
    if args.workers == 1:
        progress(map(solve, order), len(order))
        return
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('fork')) as pool:
        progress(pool.map(solve, order), len(order))


def progress(results, count):
    for done, i in enumerate(results, 1):
        if done == 1 or done % 25 == 0 or done == count:
            print(f'{done}/{count} case {i}', flush=True)


if __name__ == '__main__':
    main()
