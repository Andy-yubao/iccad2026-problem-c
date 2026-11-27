"""Run E170 S1–S4 sequentially, then evaluate the frozen S4 best."""
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import traceback

from .io import HERE, ROOT, exclusive, save, sha, sources
from .loss_config import PLAN, PROTECTED_EPOCHS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def freeze(arm, run, expected_sources):
    import torch

    result = read(run / 'results.json')
    if result['status'] not in ('early_stop', 'budget', 'max_epochs') or result['error']:
        raise RuntimeError(f'Incomplete training for {arm}')
    path = run / 'checkpoints/best.pt'
    best = torch.load(path, map_location='cpu', weights_only=False)
    progress = best['progress']
    epoch = progress['best_epoch']
    if (best['identity']['sources'] != expected_sources or sources() != expected_sources or
            best['identity']['arm'] != arm or not progress['unfrozen'] or epoch <= PROTECTED_EPOCHS or
            epoch != result['best_epoch']):
        raise ValueError(f'Invalid frozen {arm} joint best')
    candidates = [row for row in progress['history'] if row['epoch'] > PROTECTED_EPOCHS]
    winner = min(candidates, key=lambda row: row['dev_monitor']['total'])
    if winner['epoch'] != epoch or winner['dev_monitor']['total'] != result['best_fixed_loss']:
        raise ValueError(f'{arm} best differs from its fixed dev500 joint objective')
    frozen = dict(arm=arm, checkpoint=path.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(path),
                  best_epoch=epoch, parameters=sum(value.numel() for value in best['model'].values()),
                  own_selection_loss=winner['dev_monitor']['total'], selection_frozen_before_Q=True,
                  training_seconds=result['elapsed_seconds'], training_epochs=result['epochs'],
                  sources=expected_sources)
    save(HERE / f'frozen_{arm}.json', frozen)
    print(f'[LOSS] {arm} best_epoch={epoch} dev={frozen["own_selection_loss"]:.9f}', flush=True)


def stopped():
    return any((HERE / name).exists() for name in ('STOP', 'QUEUE_STOP', 'PAUSE_REQUEST'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    expected = read(HERE / 'preparation.json')['sources']
    auth = read(HERE / 'authorization.json')
    if (auth['scope'] != PLAN['revision'] or not auth['training_enabled'] or
            not auth['postprocessing_enabled'] or auth['allowed_arms'] != PLAN['stages'] or sources() != expected):
        raise ValueError('Queue authorization or frozen source identity mismatch')
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8')
    state_path = HERE / 'queue_state.json'
    with exclusive(HERE / 'queue.lock'):
        state = read(state_path)
        if args.start and (state['status'] != 'prepared' or (HERE / 'runs').exists()):
            raise RuntimeError('Fresh queue already started')
        if args.resume and state['status'] != 'paused':
            raise RuntimeError('Only a safely paused queue can resume')

        def save_state(**fields):
            state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
            save(state_path, state)

        def work(stage, module, arguments, resume=False):
            if sources() != expected:
                raise ValueError('Prepared sources changed')
            log = HERE / 'background' / f'{stage}.log'
            command = [sys.executable, '-u', '-B', '-m', f'{__package__}.{module}', *arguments]
            if resume:
                command.append('--resume')
            with log.open('a' if resume else 'x', encoding='utf-8') as sink:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8') as child:
                    save_state(active=stage, worker_pid=child.pid, log=log.relative_to(ROOT).as_posix())
                    print(f'[START] {stage} pid={child.pid} log={log.relative_to(ROOT)}', flush=True)
                    for line in child.stdout:
                        sink.write(line)
                        sink.flush()
                        print(line, end='', flush=True)
                    if child.wait():
                        raise RuntimeError(f'{stage} failed; see {log.relative_to(ROOT)}')

        def pause():
            save_state(status='paused', active=None, worker_pid=None)

        save_state(status='running', pid=os.getpid())
        print(f'[QUEUE] E170 ordered stages {", ".join(PLAN["stages"])} then dev2000', flush=True)
        try:
            for arm in PLAN['stages']:
                if stopped():
                    pause()
                    return
                if arm in state['completed']:
                    continue
                run = HERE / 'runs' / f'{arm}_seed0'
                resume = args.resume and (run / 'checkpoints/latest.pt').exists()
                if resume and read(run / 'results.json')['status'] != 'user_stop':
                    raise RuntimeError(f'{arm} lacks a safe pause result')
                work(arm, 'train', ['--arm', arm], resume)
                if read(run / 'results.json')['status'] == 'user_stop':
                    pause()
                    return
                freeze(arm, run, expected)
                state['completed'].append(arm)
                save_state(active=None, worker_pid=None)
            if stopped():
                pause()
                return
            output = HERE / 'evaluation/output'
            resume = args.resume and output.exists()
            if resume and read(output / 'state.json')['status'] != 'paused':
                raise RuntimeError('Terminal evaluation lacks a safe pause result')
            work('dev2000', 'evaluate', ['--out', str(output), '--workers',
                                        str(PLAN['terminal_evaluation']['workers'])], resume)
            if read(output / 'state.json')['status'] == 'paused':
                pause()
                return
            result = read(output / 'results.json')
            if (result['status'], result['count'], result['hard_legal']) != ('complete', 2000, 2000):
                raise RuntimeError('E170 full dev2000 hard-legality evaluation did not complete')
            state['completed'].append('dev2000')
            save_state(status='complete', active=None, worker_pid=None)
        except BaseException as error:
            save_state(status='error', error=repr(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
