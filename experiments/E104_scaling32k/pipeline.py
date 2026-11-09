"""Authorized scratch training, normal early-stop, then one frozen dev500."""
import json
import os
import subprocess
import sys
from .io import HERE,ROOT,save,exclusive

RUN=HERE/'runs/20261104_scaling128x16_seed0'

def main():
    if (HERE/'TRAINING_HOLD').exists():raise RuntimeError('User hold: preparation only; do not start')
    with exclusive(HERE/'pipeline.lock'):
        if RUN.exists():raise RuntimeError('Run exists; no automatic overwrite/restart')
        if (HERE/'STOP').exists():raise InterruptedError('STOP')
        # train creates the run itself, preserving its fresh-run identity check.
        child=subprocess.Popen([sys.executable,'-u','-m','experiments.E104_scaling32k.train'],cwd=ROOT)
        save(HERE/'pipeline_start.json',dict(status='training',controller_pid=os.getpid(),training_pid=child.pid,test_read=False))
        rc=child.wait()
        if not RUN.exists():raise RuntimeError(f'Training failed before initialization: {rc}')
        result=json.loads((RUN/'results.json').read_text(encoding='utf-8')) if (RUN/'results.json').exists() else {}
        if rc!=0 or result.get('status')!='early_stop' or (HERE/'STOP').exists() or (RUN/'STOP').exists():
            save(RUN/'pipeline_state.json',dict(status='stopped_before_evaluation',training_exit_code=rc,training=result,test_read=False))
            return
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
            env[key]='1'
        child=subprocess.Popen([sys.executable,'-u','-m','experiments.E104_scaling32k.evaluate'],cwd=ROOT,env=env)
        save(RUN/'pipeline_state.json',dict(status='evaluating',controller_pid=os.getpid(),evaluation_pid=child.pid,test_read=False))
        rc=child.wait()
        save(RUN/'pipeline_state.json',dict(status='complete' if rc==0 else 'evaluation_error',evaluation_exit_code=rc,test_read=False,next_stage='none'))
        if rc:raise RuntimeError(f'Evaluation exited {rc}; no automatic retry')

if __name__=='__main__':main()
