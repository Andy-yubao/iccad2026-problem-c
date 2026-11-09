"""Exactly E114 staged resume, then E115 first9 scratch. No old queue restart."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from .io import OWNER,ROOT,save,exclusive,sources

JOBS=(dict(experiment='E114_supervision_placement',arm='staged',resume=True,log='staged_resume'),
      dict(experiment='E115_first_head_supervision',arm='first9',resume=False,log='first9'))

def now():return datetime.now(timezone.utc).isoformat()

def main():
    with exclusive(OWNER/'queue.lock'):
        if (OWNER/'queue_state.json').exists():raise FileExistsError('Queue already dispatched; no implicit retry')
        auth=json.loads((OWNER/'authorization.json').read_text(encoding='utf-8'))
        if not auth.get('training_enabled') or not auth.get('automatic_queue'):raise PermissionError('Queue not authorized')
        gate=json.loads((OWNER/'preflight.json').read_text(encoding='utf-8'))
        if gate['status']!='passed' or gate['sources']!=sources():raise RuntimeError('E115 source gate changed')
        state=dict(status='running',pid=os.getpid(),started_at=now(),jobs=list(JOBS),completed=[],active=None,
            data='Same parent train36000 and original train/dev500 as E114 completed two arms',test_read=False,
            automatic_Q_evaluation=False,scope='Only E114 staged resume -> E115 first9 scratch')
        requested=[False];signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        child=None;folder=None
        def stopped():return requested[0] or (OWNER/'STOP').exists() or (OWNER/'QUEUE_STOP').exists()
        def beat(**kw):
            state.update(kw);state['updated_at']=now();save(OWNER/'queue_state.json',state)
        try:
            logs=OWNER/'launch_logs';logs.mkdir(exist_ok=True)
            for job in JOBS:
                if stopped():raise InterruptedError('Pair queue stopped')
                folder=ROOT/'experiments'/job['experiment']
                if (folder/'STOP').exists():raise InterruptedError('Current experiment STOP')
                args=[sys.executable,'-u','-B','-m',f'experiments.{job["experiment"]}.train','--arm',job['arm']]
                if job['resume']:args.append('--resume')
                with (logs/(job['log']+'.stdout.log')).open('x',encoding='utf-8') as out, \
                     (logs/(job['log']+'.stderr.log')).open('x',encoding='utf-8') as err:
                    child=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    beat(active=dict(**job,pid=child.pid,started_at=now()))
                    while child.poll() is None:
                        if stopped():(folder/'STOP').write_text('User stopped pair queue; preserve complete state.\n',encoding='utf-8')
                        beat();time.sleep(5)
                if child.returncode!=0:raise RuntimeError(f'{job["experiment"]} exited {child.returncode}; no next job')
                result=json.loads((folder/'runs'/f'{job["arm"]}_seed0'/'results.json').read_text(encoding='utf-8'))
                if stopped() or result['status']=='user_stop':raise InterruptedError('User stop; no next job')
                if result.get('error') or result['status'] not in ('early_stop','budget','max_epochs'):
                    raise RuntimeError('Training not normally completed; no next job')
                state['completed'].append(dict(**job,status=result['status'],epochs=result['epochs'],steps=result['steps'],completed_at=now()))
                child=None;beat(active=None)
            beat(status='complete',active=None,completed_at=now())
        except BaseException as error:
            if child is not None and child.poll() is None:
                (folder/'STOP').write_text('Pair controller exited; preserve state, no next job.\n',encoding='utf-8')
            beat(status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention',error=repr(error),traceback=traceback.format_exc())
            raise

if __name__=='__main__':main()
