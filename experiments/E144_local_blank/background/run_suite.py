"""User-authorized sequential three-experiment runner; fail/STOP closes the queue."""
import datetime
import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from ..io import ROOT,HERE,save,exclusive

DIRECTORIES=('E144_local_blank','E145_no_relation_value','E146_layout_only')
BACKGROUND=HERE/'background'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    with exclusive(BACKGROUND/'suite.lock'):
        if (BACKGROUND/'suite_state.json').exists():raise FileExistsError('No automatic restart of an existing suite')
        for directory in DIRECTORIES:
            module=importlib.import_module('experiments.'+directory+'.run_queue');module.verify()
            folder=ROOT/'experiments'/directory
            if module.blocked() or (folder/'runs').exists():raise RuntimeError('Not fresh/STOP: '+directory)
        state=dict(status='running',pid=os.getpid(),started_at=now(),order=list(DIRECTORIES),completed=[],active=None)
        requested=[False]
        signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,lambda *_:requested.__setitem__(0,True))
        def stopped():return requested[0] or (BACKGROUND/'STOP').exists() or (HERE/'STOP').exists() or (HERE/'QUEUE_STOP').exists()
        def beat(**kwargs):state.update(kwargs,updated_at=now());save(BACKGROUND/'suite_state.json',state)
        child=None
        try:
            beat()
            for directory in DIRECTORIES:
                if stopped():beat(status='paused',active=None);return
                folder=ROOT/'experiments'/directory
                flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
                logfile=BACKGROUND/(directory+'.queue.log')
                with logfile.open('x',encoding='utf-8') as sink:
                    command=[sys.executable,'-u','-B','-m','experiments.'+directory+'.run_queue','--start']
                    child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
                    active=dict(experiment=directory,pid=child.pid,log=str(logfile))
                    beat(active=active)
                    print('Starting '+directory+'; train -> frozen best -> original dev500',flush=True)
                    def pump(process):
                        for line in process.stdout:sink.write(line);sink.flush();print(line,end='',flush=True)
                    thread=threading.Thread(target=pump,args=(child,),daemon=True);thread.start()
                    while child.poll() is None:
                        if stopped() and not (folder/'STOP').exists():
                            save(folder/'STOP',dict(requested_at=now(),reason='Suite stop; block active and queued evaluations'))
                        beat();time.sleep(1)
                    thread.join();code=child.returncode;child=None
                import json
                outcome=json.loads((folder/'queue_state.json').read_text(encoding='utf-8'))
                if stopped() or outcome['status']=='paused':beat(status='paused',active=None);return
                if code!=0 or outcome['status']!='complete':raise RuntimeError(directory+' failed; later experiments not started')
                state['completed'].append(directory);beat(active=None)
            beat(status='complete',active=None,completed_at=now())
            print('All three experiments and their original dev500 validations completed.',flush=True)
        except BaseException as error:
            if child is not None and child.poll() is None:
                folder=ROOT/'experiments'/state['active']['experiment']
                if not (folder/'STOP').exists():save(folder/'STOP',dict(requested_at=now(),reason='Suite controller error'))
                child.wait()
            beat(status='needs_attention',error=repr(error),traceback=traceback.format_exc(),active=None)
            raise

if __name__=='__main__':main()
