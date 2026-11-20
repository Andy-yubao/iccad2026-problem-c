"""One dependency-bound launch: finished data -> critic training -> stop."""
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from experiments.E106_postprocessor_aware_loss.common import ROOT,OWNER,RUN,read,save,sha,sources


def main():
    with (RUN/'autostart.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0');lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        started=time.perf_counter()
        def stopped():
            return any(p.exists() for p in (OWNER/'STOP',RUN/'STOP',RUN/'BANK_ABORT',
                        OWNER/'TRAIN_STOP',RUN/'critic/STOP'))
        def beat(status,**fields):
            save(RUN/'autostart_state.json',dict(status=status,pid=os.getpid(),
                 elapsed_seconds=time.perf_counter()-started,
                 updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_read=False,
                 layout_training=False,**fields))
        try:
            if (RUN/'critic/config.json').exists():
                raise RuntimeError('Existing critic: refuse duplicate or automatic resume')
            while True:
                if stopped():
                    beat('stopped_no_training');return
                state=read(RUN/'state.json')
                if state['status'] in ('error','user_stop'):
                    raise RuntimeError('Data run did not complete normally: '+state['status'])
                if state['status']=='complete':
                    break
                beat('waiting_for_complete_data',data_progress=state)
                time.sleep(10)
            bank=read(RUN/'bank_results.json')
            assert bank['status']=='complete' and bank['cases']==5000 and bank['drafts']==25000
            assert bank['hard_legal']==25000 and bank['test_read'] is False
            assert read(RUN/'normalization.json')['fit_cases']==3000
            # Scientific package is unchanged from the running data job.
            assert bank['identity']['sources']==sources()
            auth=read(OWNER/'authorization.json')
            assert auth['critic_training'] is True and auth['layout_training'] is False
            assert auth['plan_sha256']==sha(OWNER/'plan.json')
            save(RUN/'autostart_prerequisites.json',dict(status='passed',bank_sha256=sha(RUN/'bank_results.json'),
                 normalization_sha256=sha(RUN/'normalization.json'),plan_sha256=sha(OWNER/'plan.json'),
                 complete_labels=25000,source_identity_unchanged=True,layout_training=False,test_read=False))
            env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
            def run_child(args,stage):
                if stopped():
                    beat('stopped_no_next_stage');return False
                with (RUN/'critic_stdout.log').open('ab') as stdout,(RUN/'critic_stderr.log').open('ab') as stderr:
                    child=subprocess.Popen([sys.executable,'-u','-B',*args],cwd=ROOT,env=env,
                        stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    while child.poll() is None:
                        if stopped():
                            (OWNER/'TRAIN_STOP').touch(exist_ok=True)
                        active=read(RUN/'critic/state.json') if (RUN/'critic/state.json').exists() else None
                        beat(stage,child_pid=child.pid,active_training=active)
                        time.sleep(5)
                    if child.returncode!=0:
                        raise RuntimeError(stage+' exited '+str(child.returncode)+'; see critic_stderr.log')
                if stopped():
                    beat('stopped_no_next_stage')
                    return False
                return True
            if not run_child([str(RUN/'gpu_training_gate.py')],'checking_actual_cuda_update'):
                return
            assert read(RUN/'gpu_training_gate.json')['status']=='passed'
            if not run_child(['-m','experiments.E106_postprocessor_aware_loss.train'],'critic_training'):
                return
            result=read(RUN/'critic/results.json')
            assert result['status']=='early_stop' and result['automatic_layout_training'] is False
            beat('complete_no_next_stage',result=result)
        except BaseException as exc:
            save(RUN/'autostart_error.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                 layout_training=False,test_read=False))
            beat('error_no_retry',error=repr(exc))
            raise


if __name__=='__main__':
    main()
