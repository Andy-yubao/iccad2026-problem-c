"""E3 frozen dev5 two-policy check, 90 s aggregate solver wall budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=7.5)
    ap.add_argument('--strategies', nargs='+', default=['greedy','no_soft'],
                    choices=['greedy','no_soft','imitation'])
    ap.add_argument('--checkpoint', type=Path)
    args = ap.parse_args()
    if not 0 < args.seconds <= 7.5:
        ap.error('This dev5 check caps each variant at 7.5 s, aggregate at 90 s')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from datasets.cases import load_frozen_cases
    from datasets.paths import resolved_data_paths
    from scoring.evaluation import evaluate_case, summarize_variants
    from experiments.masked_construction.construction import Optimizer
    if len(set(args.strategies))!=len(args.strategies):
        ap.error('Strategies must be unique')
    if 'imitation' in args.strategies and not args.checkpoint:
        ap.error('imitation requires --checkpoint')
    scorer=None
    if 'imitation' in args.strategies:
        from experiments.masked_construction.train import load_scorer
        scorer=load_scorer(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    files = list(Path(__file__).parent.glob('*.py'))
    write(args.output/'config.json', dict(command=subprocess.list2cmdline([sys.executable, *sys.argv]),
          utc=datetime.now(timezone.utc).isoformat(), seed=0, threads=1, device='cpu',
          strategies=args.strategies, checkpoint=str(args.checkpoint) if args.checkpoint else None,
          seconds_per_variant=args.seconds, aggregate_solver_budget_seconds=90,
          paths=resolved_data_paths(), python=sys.version, torch=torch.__version__, platform=platform.platform(),
          source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
          split_sha256=hashlib.sha256((ROOT/'protocol/split_v1.json').read_bytes()).hexdigest(),
          stage_note='raw is per-step unconstrained winner against final prefix; continuous is identity; final masks each placement',
          quality='official neutral runtime Q; actual solver time separate; not contest score'))
    result = dict(status='running', cases=[], solver_seconds=0.)
    for case in load_frozen_cases('dev'):
        record = dict(case_id=case.case_id, source=case.source, input_summary=case.input_summary(), variants={})
        result['cases'].append(record)
        for strategy in args.strategies:
            if result['solver_seconds'] >= 90:
                raise RuntimeError('Aggregate solver budget exhausted')
            solver = Optimizer(strategy, args.seconds, scorer=scorer if strategy=='imitation' else None)
            start = time.perf_counter()
            positions = solver.solve(*case.solver_inputs)
            elapsed = time.perf_counter()-start
            result['solver_seconds'] += elapsed
            final = evaluate_case(case, positions, elapsed)
            raw = evaluate_case(case, solver.raw_positions, 0.)
            final['stages'] = {'raw_proposal': raw,
                               'continuous_corrected': {'same_as': 'raw_proposal', 'correction_seconds': 0.},
                               'final': {'same_as': 'positions/metrics of this variant'}}
            final['solve_seconds'] = elapsed
            final['last_stats'] = solver.last_stats
            final['quality_change_final_minus_raw'] = final['metrics']['cost']-raw['metrics']['cost']
            record['variants'][strategy] = final
            print(case.case_id, strategy, final['metrics']['is_feasible'], final['metrics']['cost'], elapsed, flush=True)
            write(args.output/'results.json', result)
    result['summary'] = summarize_variants(result['cases'], args.strategies)
    result['status'] = 'completed'
    write(args.output/'results.json', result)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
