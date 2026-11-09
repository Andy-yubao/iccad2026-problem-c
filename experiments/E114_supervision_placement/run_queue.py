"""Run exactly the three authorized E114 training arms, sequentially, once."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from .io import OWNER, ROOT, save, exclusive, sources

ARMS = ('terminal', 'dense', 'staged')

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    with exclusive(OWNER/'queue.lock'):
        if (OWNER/'queue_state.json').exists():
            raise FileExistsError('Queue already dispatched; explicit recovery required')
        authorization = json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
        if not authorization.get('training_enabled') or not authorization.get('automatic_queue'):
            raise PermissionError('Queue has no current training authorization')
        if tuple(authorization.get('allowed_arms', [])) != ARMS:
            raise PermissionError('Queue arm scope differs from authorization')
        gate = json.loads((OWNER/'preflight.json').read_text(encoding='utf-8'))
        if gate['status'] != 'passed' or gate['sources'] != sources():
            raise ValueError('Source identity differs from implementation gate')
        state = dict(status='running', pid=os.getpid(), started_at=now(),
                     arms=list(ARMS), completed=[], active=None, test_read=False,
                     postprocessing_calls=0, automatic_evaluation=False)
        requested = [False]
        signal.signal(signal.SIGINT, lambda *_: requested.__setitem__(0, True))
        child = None
        def stopped():
            return requested[0] or (OWNER/'STOP').exists() or (OWNER/'QUEUE_STOP').exists()
        def beat(**kwargs):
            state.update(kwargs)
            state['updated_at'] = now()
            save(OWNER/'queue_state.json', state)
        try:
            for arm in ARMS:
                if stopped():
                    raise InterruptedError('Queue stopped; no next arm')
                logs = OWNER/'launch_logs'
                logs.mkdir(exist_ok=True)
                command = [sys.executable, '-u', '-B', '-m',
                           'experiments.E114_supervision_placement.train', '--arm', arm]
                with (logs/f'{arm}.stdout.log').open('x', encoding='utf-8') as out, \
                     (logs/f'{arm}.stderr.log').open('x', encoding='utf-8') as err:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    beat(active=dict(arm=arm, pid=child.pid, started_at=now()))
                    while child.poll() is None:
                        if stopped():
                            (OWNER/'STOP').write_text('Queue stopped; save complete state and do not advance.\n', encoding='utf-8')
                        beat()
                        time.sleep(5)
                result_path = OWNER/'runs'/f'{arm}_seed0'/'results.json'
                result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
                if stopped() or result.get('status') == 'user_stop':
                    raise InterruptedError('Training stopped; no next arm')
                if child.returncode != 0 or result.get('error') or result.get('status') not in ('early_stop','budget','max_epochs'):
                    raise RuntimeError(f'{arm} did not complete normally; no retry or next arm')
                state['completed'].append(dict(arm=arm, status=result['status'],
                    steps=result['steps'], epochs=result['epochs'], completed_at=now()))
                child = None
                beat(active=None)
            beat(status='complete', active=None, completed_at=now())
        except BaseException as error:
            if child is not None and child.poll() is None:
                (OWNER/'STOP').write_text('Queue exited unexpectedly; preserve state, no next arm.\n', encoding='utf-8')
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention',
                 error=repr(error), traceback=traceback.format_exc())
            raise

if __name__ == '__main__':
    main()
