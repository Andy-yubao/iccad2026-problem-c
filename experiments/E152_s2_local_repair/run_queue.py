"""Run the authorized A/B micro-tuning and common frozen dev500 evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .io import HERE, ROOT, exclusive, save, sha, sources
from .loss_config import ARMS


os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASELINE = 'e151_best45'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def verify():
    gate = read(HERE / 'preflight.json')
    if gate['status'] != 'passed' or gate['sources'] != sources():
        raise RuntimeError('E152 preparation gate or source identity mismatch')
    return gate


def freeze(arm, run, plan):
    import torch

    if arm == BASELINE:
        parent = plan['parent']
        checkpoint = ROOT / parent['checkpoint']
        expected_sha = parent['checkpoint_sha256']
    else:
        result = read(run / 'results.json')
        if (result['status'], result['epochs'], result['steps']) != ('max_epochs', plan['maximum_epochs'], plan['maximum_steps']):
            raise RuntimeError(f'{arm} did not complete the registered training schedule')
        checkpoint = run / 'checkpoints/best.pt'
        expected_sha = sha(checkpoint)
    if sha(checkpoint) != expected_sha:
        raise ValueError('Frozen checkpoint identity mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if arm != BASELINE and state['identity']['sources'] != sources():
        raise ValueError('Candidate checkpoint sources changed')
    progress = state['progress']
    best = next(row for row in progress['history'] if row['epoch'] == progress['best_epoch'])
    dev = best['dev_monitor']
    full = dev['final_total'] if arm == BASELINE else dev['original_full_L12']
    record = dict(arm=arm, checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=expected_sha,
                  best_epoch=progress['best_epoch'], own_selection_loss=dev['total'], parameters=plan['parameters'],
                  common_final_dev_loss=full, own_new_final_dev_loss=full,
                  training_seconds=state['elapsed_seconds'] if arm == BASELINE else result['elapsed_seconds'],
                  training_epochs=45 if arm == BASELINE else result['epochs'],
                  selection_frozen_before_Q=True)
    save(run / 'frozen.json', record)
    if arm != BASELINE:
        history = read(run / 'progress.json')['history']
        save(run / 'loss_comparison_points.json', dict(best=best, final_epoch=history[-1], frozen=record))
        print(f'[LOSS] arm={arm} best_epoch={record["best_epoch"]} own_dev={dev["total"]:.9f} '
              f'best_L12_overlap={dev["L12_overlap"]:.9f} final_L12_overlap={history[-1]["dev_monitor"]["L12_overlap"]:.9f}',
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--start', action='store_true')
    command.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    interrupted = [False]
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, lambda *_: interrupted.__setitem__(0, True))
    with exclusive(HERE / 'queue.lock'):
        verify()
        plan = read(HERE / 'plan.json')
        state = read(HERE / 'queue_state.json')
        markers = [HERE / name for name in ('STOP', 'QUEUE_STOP')]
        pause_file = HERE / 'PAUSE_REQUEST'
        if any(marker.exists() for marker in markers):
            raise RuntimeError('STOP present')
        if args.start:
            if state['status'] != 'prepared' or (HERE / 'runs').exists() or pause_file.exists():
                raise RuntimeError('Queue is not fresh')
            auth = read(HERE / 'authorization.json')
            if auth['training_enabled'] or auth['scope'] != plan['revision'] or auth['prepared_arms'] != list(ARMS):
                raise PermissionError('Unexpected launch authorization')
            auth.update(training_enabled=True, allowed_arms=list(ARMS), activated_at=now(),
                        preflight_sha256=sha(HERE / 'preflight.json'))
            save(HERE / 'authorization.json', auth)
            state = dict(status='running', pid=os.getpid(), queue_id=uuid.uuid4().hex,
                         started_at=now(), sessions=1, completed=[], active=None)
        else:
            if state['status'] != 'paused':
                raise RuntimeError('Only a safely paused queue can resume')
            if pause_file.exists():
                if read(pause_file)['queue_id'] != state['queue_id']:
                    raise ValueError('Unrelated pause marker')
                destination = HERE / 'pause_history' / (uuid.uuid4().hex + '.json')
                destination.parent.mkdir(exist_ok=True)
                pause_file.rename(destination)
            state.update(status='running', pid=os.getpid(), sessions=state['sessions'] + 1)

        def beat(**updates):
            state.update(updates)
            save(HERE / 'queue_state.json', dict(state, updated_at=now()))

        def pause_requested():
            if interrupted[0] or any(marker.exists() for marker in markers):
                if not pause_file.exists():
                    save(pause_file, dict(queue_id=state['queue_id'], requested_at=now()))
            return pause_file.exists()

        child = None
        phases = [(arm, phase) for arm in ARMS for phase in ('train', 'freeze')]
        phases += [(BASELINE, 'freeze'), *((arm, 'evaluation') for arm in (*ARMS, BASELINE))]
        try:
            beat()
            for arm, phase in phases:
                key = f'{arm}/{phase}'
                if key in state['completed']:
                    continue
                if pause_requested():
                    beat(status='paused', active=None)
                    return
                run = HERE / 'runs' / f'{arm}_seed0'
                if phase == 'freeze':
                    freeze(arm, run, plan)
                else:
                    is_train = phase == 'train'
                    module = '.train' if is_train else '.evaluation.run'
                    command = [sys.executable, '-u', '-B', '-m', 'experiments.E152_s2_local_repair' + module,
                               '--arm', arm]
                    existing = run if is_train else run / 'evaluation_no_mib_dev500'
                    if existing.exists():
                        command.append('--resume')
                    logs = HERE / 'launch_logs'
                    logs.mkdir(exist_ok=True)
                    log = logs / f'{arm}_{phase}_{state["sessions"]:03d}.log'
                    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                    with log.open('x', encoding='utf-8') as sink:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                 text=True, encoding='utf-8', errors='replace', creationflags=flags)

                        def pump():
                            for line in child.stdout:
                                sink.write(line)
                                sink.flush()
                                print(line, end='', flush=True)

                        thread = threading.Thread(target=pump, daemon=True)
                        thread.start()
                        beat(active=dict(arm=arm, phase=phase, pid=child.pid, log=log.relative_to(ROOT).as_posix()))
                        while child.poll() is None:
                            pause_requested()
                            beat()
                            time.sleep(.5)
                        thread.join()
                    code = child.returncode
                    child = None
                    if code:
                        raise RuntimeError(f'{key} exited with {code}. See {log.relative_to(ROOT)}')
                    result = read(run / ('results.json' if is_train else 'evaluation_no_mib_dev500/state.json'))
                    if result['status'] in ('user_stop', 'paused'):
                        beat(status='paused', active=None)
                        return
                    if result['status'] != ('max_epochs' if is_train else 'complete'):
                        raise RuntimeError(f'{key} ended in {result["status"]}')
                state['completed'].append(key)
                beat(active=None)
            from .comparison import summarize
            summarize()
            beat(status='complete', active=None, completed_at=now())
        except BaseException as error:
            if child is not None and child.poll() is None:
                save(pause_file, dict(queue_id=state['queue_id'], requested_at=now()))
                child.wait()
            beat(status='needs_attention', active=None, error=repr(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
