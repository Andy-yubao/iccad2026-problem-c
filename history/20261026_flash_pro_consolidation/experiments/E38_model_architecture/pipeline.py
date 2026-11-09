"""User-started sequential capacity study; explicit resume and safe stage stops."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from .arms import ARMS
from .io import ROOT, HERE, save, sources, exclusive


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=('all', *ARMS), default='all')
    ap.add_argument('--tag', default='20261022_capacity')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-hours', type=float, default=4.)
    ap.add_argument('--max-epochs', type=int, default=200)
    ap.add_argument('--stage', choices=('all', 'train', 'final'), default='all')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('Tag must contain only letters, digits, underscores or hyphens')
    if not 0 < args.max_hours <= 4 or not 1 <= args.max_epochs <= 200:
        raise ValueError('Positive training budget and 1..200 epochs required')
    folder = HERE/'runs'/f'{args.tag}_pipeline_seed{args.seed}'
    identity = {'sources': sources(), 'arm': args.arm, 'seed': args.seed,
                'max_hours': args.max_hours, 'max_epochs': args.max_epochs,
                'batch_size': 32, 'test_read': False}
    with exclusive(folder.with_suffix('.lock')):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'config.json'
        if path.exists():
            if not args.resume or json.loads(path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Existing pipeline requires explicit --resume and identical source/budget identity')
        else:
            save(path, identity)
        if (HERE/'runs/STOP').exists() or (folder/'STOP').exists():
            print('Remove the experiment/pipeline STOP file before resuming', flush=True)
            return
        history_path = folder/'progress.json'
        history = json.loads(history_path.read_text(encoding='utf-8'))['stages'] if history_path.exists() else []
        requested = [False]
        previous = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        def stopped():
            return requested[0] or (folder/'STOP').exists() or (HERE/'runs/STOP').exists()
        def stage(module, arguments, output, kind):
            if stopped():
                return False
            if (output/'STOP').exists():
                print('Paused by STOP:', output, flush=True)
                return False
            command = [sys.executable, '-u', '-B', '-m', f'experiments.E38_model_architecture.{module}', *arguments]
            start = time.perf_counter()
            environment = dict(os.environ, E38_THREADS='4')
            with (folder/'console.log').open('a', encoding='utf-8') as log:
                log.write('\n'+json.dumps({'command': command, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})+'\n')
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                           creationflags=creationflags)
                save(folder/'active.json', {'pid': process.pid, 'kind': kind, 'command': command,
                                           'output': output.relative_to(ROOT).as_posix()})
                # Drain stdout on a helper thread; poll independently so quiet children still see STOP.
                import threading
                def drain():
                    for line in process.stdout:
                        sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
                reader = threading.Thread(target=drain)
                reader.start()
                while process.poll() is None:
                    if stopped():
                        output.mkdir(parents=True, exist_ok=True); (output/'STOP').touch()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                reader.join()
            history.append({'kind': kind, 'command': command, 'returncode': process.returncode,
                            'seconds': time.perf_counter()-start, 'output': output.relative_to(ROOT).as_posix()})
            save(history_path, {'stages': history, 'test_read': False})
            (folder/'active.json').unlink(missing_ok=True)
            if process.returncode:
                raise RuntimeError(f'{kind} failed with exit code {process.returncode}; see {folder / "console.log"}')
            return not stopped() and not (output/'STOP').exists()
        try:
            arms = list(ARMS) if args.arm == 'all' else [args.arm]
            # E25 reference is measured through this same copied solver and CPU resource configuration.
            reference = HERE/'runs'/f'{args.tag}_e25_reference'
            if args.stage in ('all', 'final'):
                if not stage('final', ['--output', str(reference), '--baseline'], reference, 'E25 reference'):
                    return
            for arm in arms:
                output = HERE/'runs'/f'{args.tag}_{arm}_seed{args.seed}'
                if args.stage in ('all', 'train'):
                    command = ['--arm', arm, '--output', str(output), '--seed', str(args.seed),
                               '--max-hours', str(args.max_hours), '--max-epochs', str(args.max_epochs), '--batch-size', '32']
                    if args.resume and (output/'checkpoints/latest.pt').exists():
                        command.append('--resume')
                    if not stage('train', command, output, 'train '+arm):
                        return
                    result = json.loads((output/'results.json').read_text(encoding='utf-8'))
                    if result['status'] not in ('early_stop', 'budget', 'max_epochs'):
                        return
                if args.stage in ('all', 'final'):
                    if not stage('final', ['--output', str(output)], output, 'final '+arm):
                        return
                from .analyze import analyze
                analyze(args.tag, args.seed)
            save(folder/'results.json', {'status': 'complete', 'stage': args.stage, 'stages': history,
                                        'arms': arms, 'test_read': False, 'baseline': reference.relative_to(ROOT).as_posix()})
        finally:
            signal.signal(signal.SIGINT, previous)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
