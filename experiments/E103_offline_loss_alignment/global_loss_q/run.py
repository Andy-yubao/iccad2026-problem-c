"""Fresh unpaired cross-problem data -> global statistics -> comparison figures."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
import zipfile
import torch
from ..cache import load_rows
from .common import ROOT,OWNER,RUN,read,save,sha,setup,check,sources
from .worker import initialize,evaluate

def main():
    setup();plan=read(OWNER/'plan.json');panel=read(OWNER/'panel.json');gate=read(OWNER/'preflight.json');current=sources()
    assert gate['status']=='passed' and gate['sources']==current
    identity=gate['identity']
    if (RUN/'config.json').exists():raise RuntimeError('Run already started; no duplicate or blind retry')
    RUN.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (RUN/'pipeline.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        save(RUN/'config.json',dict(plan=plan,identity=identity))
        with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in current:z.write(ROOT/name,name)
        started=time.perf_counter();hb=dict(status='running',stage='preparing_new_train2000',completed=0,total=2000,workers=4,test_read=False)
        closed=threading.Event();mutex=threading.RLock()
        def beat(**fields):
            with mutex:
                hb.update(fields);save(RUN/'state.json',dict(**hb,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeat():
            while not closed.wait(5):beat()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        def stop(*_):(RUN/'STOP').write_text('User stop; no following cases or analysis.\n',encoding='utf-8')
        previous=signal.signal(signal.SIGINT,stop)
        try:
            missing_inputs=[i for i in range(2000) if not (RUN/f'inputs/{i:04d}.pt').exists()]
            rows=load_rows('train',[panel['samples'][i]['train_index'] for i in missing_inputs])
            for i,row in zip(missing_inputs,rows):
                check();assert list(row['member'])==panel['samples'][i]['member']
                path=RUN/f'inputs/{i:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(row,path)
            del rows
            remaining=[i for i in range(2000) if not (RUN/f'observations/{i:04d}.json').exists()]
            completed=2000-len(remaining);beat(stage='fresh_global_loss_Q',completed=completed)
            iterator=iter(remaining)
            with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(identity,)) as pool:
                pending={}
                for _ in range(4):
                    i=next(iterator,None)
                    if i is not None:pending[pool.submit(evaluate,i)]=i
                while pending:
                    check();ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                    for future in ready:
                        pending.pop(future);future.result();completed+=1;beat(completed=completed)
                        if completed%25==0:print('new independent observations',completed,'/2000',flush=True)
                        check();j=next(iterator,None)
                        if j is not None:pending[pool.submit(evaluate,j)]=j
            check();beat(stage='global_statistics_and_comparison_figures',completed=2000)
            from .analysis import main as analyze
            result=analyze()
            for spec in plan['models']:assert sha(ROOT/spec['checkpoint'])==spec['sha256']
            beat(status='complete',stage='global_analysis_and_figures_ready')
            print('Complete',result['comparisons'],flush=True)
        except InterruptedError:beat(status='user_stop',stage='stopped_no_next_stage')
        except BaseException as exc:
            save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry');raise
        finally:closed.set();thread.join(timeout=6);signal.signal(signal.SIGINT,previous)

if __name__=='__main__':main()
