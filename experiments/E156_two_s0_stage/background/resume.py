"""Resume the unchanged frozen S0 run, then freeze its own-Loss best only."""
import json
import os
import subprocess
import sys
from ..io import HERE,ROOT,save,sources,exclusive
from ..run_queue import freeze


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert read(HERE/'preflight.json')['sources']==sources()
    with exclusive(HERE/'queue.lock'):
        state=read(HERE/'queue_state.json')
        assert state['status']=='stopped' and not (HERE/'PAUSE_REQUEST').exists() and not (HERE/'QUEUE_STOP').exists()
        state.update(status='running',active='e156_s0_backend',pid=os.getpid(),resumed=True)
        save(HERE/'queue_state.json',state)
        code=subprocess.call([sys.executable,'-u','-B','-m','experiments.E156_two_s0_stage.train',
            '--arm','e156_s0_backend','--resume'],cwd=ROOT)
        result=read(HERE/'runs/e156_s0_backend_seed0/results.json')
        if code or result['error']:
            state.update(status='needs_attention',active=None)
        elif result['status']=='user_stop':
            state.update(status='stopped',active=None)
        else:
            freeze('e156_s0_backend')
            state.update(status='complete',active=None,completed=['e156_s0_backend'],S1_S2_training_started=False,postprocessing_calls=0)
        save(HERE/'queue_state.json',state)


if __name__=='__main__':main()
