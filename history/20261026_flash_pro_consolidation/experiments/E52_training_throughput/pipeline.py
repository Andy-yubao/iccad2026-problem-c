"""User-started E52; default trains exactly one arm and exits."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from .io import HERE, ROOT, save, sources, exclusive, sha
from .module_loss import definition

CONTROL = ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'


def folders(arm, engine='cuda_graph'):
    name = f'20261025_rectangle_w96d12_{engine}_{arm}_seed0'
    return HERE/'runs'/name, HERE/'runs'/(name+'_pipeline')


def commands(arm='main', resume=False, engine='cuda_graph'):
    output, _ = folders(arm,engine)
    train = [sys.executable, '-u', '-B', '-m', 'experiments.E52_training_throughput.train',
             '--arm', arm, '--output', str(output), '--device', 'cuda', '--seed', '0',
             '--max-hours', '4', '--max-epochs', '200', '--batch-size', '32', '--scheduler-start-epoch', '15',
             '--engine',engine]
    if resume and (output/'checkpoints/latest.pt').exists():
        train.append('--resume')
    return {'train': train,
            'final': [sys.executable, '-u', '-B', '-m', 'experiments.E52_training_throughput.final', '--output', str(output)],
            'report': [sys.executable, '-u', '-B', '-m', 'experiments.E52_training_throughput.report', '--output', str(output)],
            'review': [sys.executable, '-u', '-B', '-m', 'experiments.E52_training_throughput.prepare_review', '--output', str(output)]}


def control_identity():
    frozen = json.loads((CONTROL/'frozen.json').read_text(encoding='utf-8'))
    if sha(CONTROL/'checkpoints/best.pt') != frozen['checkpoint_sha256']:
        raise ValueError('Frozen E42 checkpoint changed')
    for name, digest in frozen['sources'].items():
        if name.startswith('experiments/E42_joint_regularization/') and sha(ROOT/name) != digest:
            raise ValueError('Frozen E42 algorithm changed: '+name)
    return {name: sha(CONTROL/name) for name in ('frozen.json', 'full_dev.json', 'terminal/results.json', 'checkpoints/best.pt')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=('train', 'final', 'all', 'review'), default='train')
    ap.add_argument('--arm', choices=('main',), default='main')
    ap.add_argument('--engine', choices=('cuda_graph','resident_eager','reference'), default='cuda_graph')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    output, coordinator = folders(args.arm,args.engine)
    kinds = {'train': ['train'], 'final': ['final', 'report', 'review'],
             'all': ['train', 'final', 'report', 'review'], 'review': ['review']}[args.stage]
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'arm': args.arm, 'stage': args.stage,
                          'commands': {k: commands(args.arm, args.resume,args.engine)[k] for k in kinds},
                          'output': str(output), 'automatic_other_training': False,
                          'training_started': False, 'evaluation_started': False}, indent=2))
        return
    identity = {'sources': sources(), 'control': control_identity(), 'arm': args.arm, 'engine':args.engine,
                'seed': 0, 'device': 'cuda', 'max_seconds': 14400, 'max_epochs': 200,
                'batch_size': 32, 'cpu_threads': 4, 'module_loss': definition(), 'test_read': False}
    with exclusive(coordinator.with_suffix('.lock')):
        coordinator.mkdir(parents=True, exist_ok=True)
        config = coordinator/'config.json'
        if config.exists():
            if not args.resume or json.loads(config.read_text(encoding='utf-8')) != identity:
                raise ValueError('Use -Resume with identical sources/data/control/budget')
        else:
            save(config, identity)
        requested = [False]
        flags = [HERE/'runs/STOP', coordinator/'STOP', output/'STOP']
        def stopped():
            return requested[0] or any(p.exists() for p in flags)
        if stopped():
            print('STOP present; remove only for an explicit resume.'); return
        old = signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        history_path = coordinator/'progress.json'
        history = json.loads(history_path.read_text(encoding='utf-8'))['stages'] if history_path.exists() else []
        try:
            for kind in kinds:
                if stopped(): return
                argv = commands(args.arm, args.resume,args.engine)[kind]
                started = time.perf_counter()
                env = dict(os.environ, E52_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', NUMBA_NUM_THREADS='2', MPLBACKEND='Agg')
                with (coordinator/'console.log').open('a', encoding='utf-8') as log:
                    log.write(json.dumps({'kind': kind, 'argv': argv, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})+'\n'); log.flush()
                    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    save(coordinator/'active.json', {'kind': kind, 'pid': process.pid, 'argv': argv})
                    def drain():
                        for line in process.stdout:
                            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
                    reader = threading.Thread(target=drain); reader.start()
                    while process.poll() is None:
                        if stopped(): (HERE/'runs/STOP').touch()
                        try: process.wait(timeout=1)
                        except subprocess.TimeoutExpired: pass
                    reader.join()
                history.append({'kind': kind, 'argv': argv, 'returncode': process.returncode,
                                'seconds': time.perf_counter()-started})
                save(history_path, {'stages': history, 'test_read': False})
                (coordinator/'active.json').unlink(missing_ok=True)
                if process.returncode:
                    raise RuntimeError(f'{kind} failed; completed evidence retained in {coordinator}')
                if stopped(): return
                if kind == 'train':
                    result = json.loads((output/'results.json').read_text(encoding='utf-8'))
                    if result['status'] not in ('early_stop', 'budget', 'max_epochs'): return
            save(coordinator/'results.json', {'status': 'complete', 'stage': args.stage,
                'stages': history, 'training_arms': [args.arm], 'test_read': False, 'further_training_jobs': []})
        finally:
            signal.signal(signal.SIGINT, old)


if __name__ == '__main__':
    main()
