"""Reuse enhanced1000, complete current1000 with4 workers, fit one coefficient vector."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
import zipfile
from .common import ROOT,OWNER,RUN,BANK,KEYS,read,save,sha,setup,check,sources
from .worker import initialize,evaluate_case

def main():
    setup();gate=read(OWNER/'preflight.json');current=sources()
    assert gate['status']=='passed' and gate['sources']==current
    identity=gate['identity'];plan=read(OWNER/'plan.json')
    if (RUN/'config.json').exists():raise RuntimeError('Already started; explicit review required before resume')
    if (RUN/'error.json').exists() or (RUN/'results.json').exists():raise RuntimeError('Existing error/completion; no automatic restart')
    assert sha(ROOT/plan['current_checkpoint'])==plan['current_sha256']
    RUN.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (RUN/'pipeline.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        save(RUN/'config.json',dict(plan=plan,identity=identity))
        with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in current:z.write(ROOT/name,name)
        started=time.perf_counter();closed=threading.Event();mutex=threading.RLock()
        hb=dict(status='running',stage='validating_enhanced_cache',total=1000,completed=0,workers=4,test_read=False)
        def beat(**fields):
            with mutex:
                hb.update(fields);save(RUN/'state.json',dict(**hb,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeat():
            while not closed.wait(5):beat()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        def stop(*_):(RUN/'STOP').write_text('User stop; no following cases or fit.\n',encoding='utf-8')
        previous=signal.signal(signal.SIGINT,stop)
        try:
            panel=read(OWNER.parent/'panel.json')
            audited={r['path']:r['sha256'] for r in read(BANK/'audits/q_sanity.json')['bank_hashes']}
            reused=[]
            for i in range(1000):
                check();path=BANK/f'bank/{i:04d}.json';assert sha(path)==audited[path.relative_to(ROOT).as_posix()]
                cached=read(path);parent=next(d for d in cached['drafts'] if d['model_id']=='e98_generalization')
                assert parent['checkpoint_sha256']==plan['enhanced_sha256'] and cached['member']==panel['members'][i]
                assert set(parent['loss_terms'])==set(KEYS)
                reused.append(dict(index=i,member=cached['member'],path=path.relative_to(ROOT).as_posix(),
                    model_id='e98_generalization',source_sha256=sha(path)))
            save(RUN/'enhanced_cache_reuse.json',dict(count=1000,records=reused,bridge_preflight=OWNER.relative_to(ROOT).as_posix()+'/preflight.json'))
            missing=[i for i in range(1000) if not (RUN/f'current/{i:04d}.json').exists()]
            complete=1000-len(missing)
            beat(stage='current_train1000',completed=complete)
            print('Reused enhanced1000; preserved current',complete,'remaining',len(missing),flush=True)
            iterator=iter(missing)
            with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(identity,)) as pool:
                pending={}
                for _ in range(4):
                    i=next(iterator,None)
                    if i is not None:pending[pool.submit(evaluate_case,i)]=i
                while pending:
                    check();ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                    for future in ready:
                        i=pending.pop(future);future.result();complete+=1;beat(completed=complete)
                        if complete%10==0:print('current train',complete,'/1000',flush=True)
                        check();j=next(iterator,None)
                        if j is not None:pending[pool.submit(evaluate_case,j)]=j
            check();beat(stage='one_fixed_lambda_weight_fit',completed=1000)
            from .fitting import main as fit_once
            result=fit_once()
            assert sha(ROOT/plan['current_checkpoint'])==plan['current_sha256']
            beat(status='complete',stage='weight_proposal_ready')
            print('Completed one two-source fit; informative fit pairs',result['proposal']['pair_count'],flush=True)
        except InterruptedError:beat(status='user_stop',stage='stopped_no_next_stage')
        except BaseException as exc:
            save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry');raise
        finally:closed.set();thread.join(timeout=6);signal.signal(signal.SIGINT,previous)

if __name__=='__main__':main()
