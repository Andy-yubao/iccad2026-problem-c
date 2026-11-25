"""Train E169 S2 and export its frozen best."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from .io import HERE, ROOT, exclusive, save, sha, sources
from .loss_config import ARM, PROTECTED_EPOCHS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def freeze(arm, source_identity):
    import torch

    run = HERE / 'runs' / f'{arm}_seed0'
    result = read(run / 'results.json')
    if result['status'] not in ('early_stop', 'budget', 'max_epochs') or result['error']:
        raise RuntimeError(f'Incomplete training for {arm}')
    path = run / 'checkpoints' / 'best.pt'
    best = torch.load(path, map_location='cpu', weights_only=False)
    if best['identity']['sources'] != source_identity or source_identity != sources():
        raise ValueError('Frozen source identity changed')
    progress = best['progress']
    epoch = progress['best_epoch']
    row = next(row['dev_monitor'] for row in progress['history'] if row['epoch'] == epoch)
    if epoch <= PROTECTED_EPOCHS or not progress['unfrozen']:
        raise ValueError('Best must come from a completed joint epoch')
    frozen = dict(arm=arm, checkpoint=path.relative_to(ROOT).as_posix(),
                  checkpoint_sha256=sha(path), best_epoch=epoch,
                  parameters=sum(value.numel() for value in best['model'].values()),
                  own_selection_loss=row['total'], training_seconds=result['elapsed_seconds'],
                  training_epochs=result['epochs'], selection_frozen_before_Q=True)
    frozen['common_final_dev_loss'] = row.get('final_base_full_total', row['total'])
    frozen['own_new_final_dev_loss'] = row.get('final_total', row['total'])
    save(HERE / f'frozen_{arm}.json', frozen)
    print(f'[LOSS] {arm} best_epoch={epoch} dev={row["total"]:.9f}', flush=True)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    expected = read(HERE / 'preparation.json')['sources']
    auth = read(HERE / 'authorization.json')
    if not auth['training_enabled'] or ARM not in auth['allowed_arms'] or expected != sources():
        raise ValueError('Queue authorization or frozen sources mismatch')
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8')
    state_path = HERE / 'queue_state.json'
    with exclusive(HERE / 'queue.lock'):
        state = read(state_path)
        if args.start and (state['status'] != 'prepared' or (HERE / 'runs').exists()):
            raise RuntimeError('Fresh queue already started')
        if args.resume and state['status'] != 'paused':
            raise RuntimeError('Only a safely paused queue can resume')

        def beat(**fields):
            state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
            save(state_path, state)

        def stopped():
            return any((HERE / name).exists() for name in ('STOP', 'QUEUE_STOP', 'PAUSE_REQUEST'))

        def run(stage, module, arm, resume=False):
            if expected != sources():
                raise ValueError('Prepared source identity changed')
            log = HERE / 'background' / f'{stage}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a' if resume else 'x', encoding='utf-8') as sink:
                command = [sys.executable, '-u', '-B', '-m', f'{__package__}.{module}', '--arm', arm]
                if resume:
                    command.append('--resume')
                    sink.write(f'\n[RESUME] {datetime.now(timezone.utc).isoformat()}\n')
                    sink.flush()
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8') as child:
                    beat(active=stage, worker_pid=child.pid, log=log.relative_to(ROOT).as_posix())
                    print(f'[START] {stage} pid={child.pid} log={log.relative_to(ROOT)}', flush=True)
                    for line in child.stdout:
                        sink.write(line)
                        sink.flush()
                        print(line, end='', flush=True)
                    if child.wait():
                        raise RuntimeError(f'{stage} failed; see {log.relative_to(ROOT)}')

        beat(status='running', pid=os.getpid())
        try:
            if stopped():
                beat(status='paused', active=None, worker_pid=None)
                return
            run_dir = HERE / 'runs' / f'{ARM}_seed0'
            resume = args.resume and (run_dir / 'checkpoints/latest.pt').exists()
            if resume and read(run_dir / 'results.json')['status'] != 'user_stop':
                raise RuntimeError('S2 lacks a safe pause result')
            run(ARM, 'train', ARM, resume=resume)
            if read(run_dir / 'results.json')['status'] == 'user_stop':
                beat(status='paused', active=None, worker_pid=None)
                return
            freeze(ARM, expected)
            beat(status='complete', active=None, worker_pid=None, completed=[ARM])
        except BaseException as error:
            beat(status='error', error=repr(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
