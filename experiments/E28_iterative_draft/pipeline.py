"""User-started pipeline; default runs the main arm, matrix requires --arm all."""
import argparse
import json
import os
import re
import subprocess
import sys
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
from .io import ROOT, HERE, save, exclusive
from .model import ARMS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=(*ARMS, 'all'), default='recurrent_full')
    ap.add_argument('--stage', choices=('prepare', 'train', 'final', 'all'), default='all')
    ap.add_argument('--tag', default='20261021')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-hours', type=float, default=4.)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.tag) or min(args.max_hours, args.max_epochs) <= 0:
        raise ValueError('Use a simple tag and positive budgets')
    log = HERE / 'runs' / (args.tag + '_pipeline_' + args.arm)
    def stopped():
        return (HERE / 'runs/STOP').exists()
    def stage(name, command):
        save(log / 'progress.json', {'stage': name, 'status': 'running', 'command': command, 'test_read': False})
        print('E28 stage:', name, flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException as error:
            save(log / 'progress.json', {'stage': name, 'status': 'interrupted_or_failed', 'error': repr(error), 'test_read': False})
            raise
    with exclusive(log.with_suffix('.lock')):
        if stopped():
            print('E28 global STOP exists; no stage started', flush=True); return
        prefix = [sys.executable, '-u', '-B', '-m']
        if args.stage in ('prepare', 'all'):
            stage('prepare', prefix + ['experiments.E28_iterative_draft.prepare', '--device', args.device])
            from .data import CACHE
            if stopped() or (CACHE / 'STOP').exists() or not (CACHE / 'results.json').exists():
                save(log / 'progress.json', {'stage': 'prepare', 'status': 'paused', 'test_read': False}); return
        arms = list(ARMS) if args.arm == 'all' else [args.arm]
        for arm in arms:
            output = HERE / 'runs' / f'{args.tag}_{arm}_v5_seed0'
            if stopped() or (output / 'STOP').exists():
                save(log / 'progress.json', {'stage': arm, 'status': 'paused', 'test_read': False}); return
            if args.stage in ('train', 'all'):
                command = prefix + ['experiments.E28_iterative_draft.train', '--arm', arm, '--output', str(output),
                                     '--device', args.device, '--max-hours', str(args.max_hours),
                                     '--max-epochs', str(args.max_epochs), '--batch-size', '32']
                if args.resume and (output / 'checkpoints/latest.pt').exists():
                    command += ['--resume']
                stage('train_' + arm, command)
                progress = json.loads((output / 'progress.json').read_text(encoding='utf-8'))
                if not progress['finished'] or stopped() or (output / 'STOP').exists():
                    save(log / 'progress.json', {'stage': arm, 'status': 'paused', 'test_read': False}); return
            if args.stage in ('final', 'all'):
                command = prefix + ['experiments.E28_iterative_draft.final_evaluate', '--run', str(output),
                                     '--device', args.device, '--resume']
                stage('raw_final_' + arm, command)
                if not (output / 'final_evaluation/results.json').exists() or not (output / 'full_dev.json').exists():
                    save(log / 'progress.json', {'stage': arm, 'status': 'paused', 'test_read': False}); return
        save(log / 'progress.json', {'stage': args.stage, 'arms': arms, 'status': 'complete', 'test_read': False})


if __name__ == '__main__':
    main()
