"""Explicitly requested interruption recovery; keep frozen training sources intact."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.E104_scaling32k.io import HERE,save,exclusive

RUN=HERE/'runs/20261104_scaling128x16_seed0'

def stopped():
    return any(p.exists() for p in (HERE/'STOP',RUN/'STOP',HERE/'TRAINING_HOLD'))

def main():
    with exclusive(HERE/'pipeline.lock'):
        if stopped():raise InterruptedError('Stop or hold present')
        child=subprocess.Popen([sys.executable,'-u','-m','experiments.E104_scaling32k.train','--resume'],cwd=ROOT)
        save(RUN/'pipeline_state.json',dict(status='resuming_training',controller_pid=os.getpid(),training_pid=child.pid,explicit_user_restart=True,test_read=False))
        code=child.wait()
        result=json.loads((RUN/'results.json').read_text(encoding='utf-8')) if (RUN/'results.json').exists() else {}
        if code!=0 or result.get('status')!='early_stop' or stopped():
            save(RUN/'pipeline_state.json',dict(status='stopped_before_evaluation',training_exit_code=code,training=result,test_read=False,no_automatic_retry=True))
            return
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
            env[key]='1'
        child=subprocess.Popen([sys.executable,'-u','-m','experiments.E104_scaling32k.evaluate'],cwd=ROOT,env=env)
        save(RUN/'pipeline_state.json',dict(status='evaluating',controller_pid=os.getpid(),evaluation_pid=child.pid,test_read=False))
        code=child.wait()
        save(RUN/'pipeline_state.json',dict(status='complete' if code==0 else 'evaluation_error',evaluation_exit_code=code,test_read=False,next_stage='none'))
        if code:raise RuntimeError(f'Evaluation exit {code}; no automatic retry')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(Path(__file__).with_name('controller_error.json'),dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
        raise
