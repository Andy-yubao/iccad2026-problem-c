"""Authorized 2026-11-08 dev500 evaluation only; stop on any failure."""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from ..io import ROOT, save, exclusive

OWNER = Path(__file__).resolve().parent
JOBS = [
    ('E114_supervision_placement', 'evaluation_staged', 'staged'),
    ('E115_first_head_supervision', 'evaluation', 'first9'),
]


def main():
    state = dict(status='running', pid=os.getpid(), completed=[], active=None,
                 test_read=False, training=False, automatic_retry=False,
                 started_at=datetime.now(timezone.utc).isoformat())
    with exclusive(OWNER / 'queue.lock'):
        try:
            for experiment, folder, arm in JOBS:
                if (OWNER / 'STOP').exists():
                    raise InterruptedError('Evaluation queue STOP; no next job')
                state['active'] = dict(experiment=experiment, arm=arm)
                save(OWNER / 'queue_state.json', state)
                with (OWNER / f'{arm}.stdout.log').open('w', encoding='utf-8') as stdout, \
                     (OWNER / f'{arm}.stderr.log').open('w', encoding='utf-8') as stderr:
                    child = subprocess.Popen(
                        [sys.executable, '-u', '-B', '-m', f'experiments.{experiment}.{folder}.run', '--arm', arm],
                        cwd=ROOT, stdout=stdout, stderr=stderr)
                    state['active']['pid'] = child.pid
                    save(OWNER / 'queue_state.json', state)
                    while True:
                        try:
                            code = child.wait(timeout=2)
                            break
                        except subprocess.TimeoutExpired:
                            if (OWNER / 'STOP').exists():
                                save(ROOT / 'experiments' / experiment / 'runs' / f'{arm}_seed0' /
                                     'evaluation_no_mib_dev500' / 'STOP', {'source': 'pair evaluation STOP'})
                    if code:
                        raise RuntimeError(f'{arm} exited {code}; no retry or next model')
                state['completed'].append(state['active'])
                state['active'] = None
                save(OWNER / 'queue_state.json', state)
            state['status'] = 'complete'
        except BaseException as error:
            state.update(status='user_stop' if isinstance(error, InterruptedError) else 'error', error=repr(error))
            raise
        finally:
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            save(OWNER / 'queue_state.json', state)


if __name__ == '__main__':
    main()
